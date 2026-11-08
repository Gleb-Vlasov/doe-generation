"""Обучение UnrolledDOENet под метрику оценки (амплитуды, нормировка на максимум, ИСС + КК).

Обучающие данные генерируются на лету (символы, фигуры, кольца, линии, полутона) + фрагменты фотографий.
    python train.py --run runs/main --steps 30000
    python train.py --run runs/smoke --steps 50 --eval-every 25 --depth 4      # проверка за 1-2 минуты
Необязательно: --val <папка с четвёрками для отчёта>, --photos <папки с jpg/png для обучения>,
--photo-test <папка фото для контроля>. Повторный запуск с тем же --run продолжает обучение.
Итоговые веса: <run>/best.pt -> скопировать в weights/model.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doe import core as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE


@torch.no_grad()
def evaluate(model, codes, device, batch=32, domain="amp"):
    model.eval()
    ps, ss = [], []
    for i in range(0, len(codes), batch):
        t = C.to_target(codes[i:i + batch], device)
        phase = model(t)
        phase = C.codes_to_phase(C.phase_to_codes(phase))           # ровно как после записи PNG
        amp = C.recon(C.propagate(phase, model.chirp), domain)
        amp8 = torch.round(C.norm_max(amp) * 255)                   # как сохранённый 8-битный PNG
        p, s = C.scores(amp8, t)
        ps.append(p.cpu()); ss.append(s.cpu())
    model.train()
    p, s = torch.cat(ps), torch.cat(ss)
    return float(p.mean()), float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "runs", "main"))
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.3, help="вес КК в лоссе; 1-alpha — вес ИСС")
    ap.add_argument("--domain", default="amp", choices=["amp", "int", "both"],
                    help="amp — амплитуды (текущее правило), int — интенсивности, both — страховка")
    ap.add_argument("--p-photo", type=float, default=0.25)
    ap.add_argument("--p-gray", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--refine", type=int, default=48, help="число слоёв уточнения по градиенту метрики после физических слоёв")
    ap.add_argument("--init", default="", help="начальные веса (best.pt предыдущего обучения)")
    ap.add_argument("--hours", type=float, default=0, help="бюджет времени обучения в часах (вместо --steps)")
    ap.add_argument("--amp", action="store_true", help="смешанная точность fp16 (быстрее, но градиенты физических слоёв переполняются; по умолчанию выкл.)")
    ap.add_argument("--val", default=os.path.join(ROOT, "data", "val"))
    ap.add_argument("--photos", nargs="*", default=[os.path.join(ROOT, "data", "photos", d) for d in ("train", "val")])
    ap.add_argument("--photo-test", default=os.path.join(ROOT, "data", "photos", "test"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    os.makedirs(args.run, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    photos = C.PhotoBank([p for p in args.photos if os.path.isdir(p)])
    print("фото для обучения:", len(photos), "| шрифты:", [os.path.basename(p) for p in C.FONT_PATHS])

    # Наборы для контроля: независимый синтетический (выбор лучшей модели) и dataset_v1/val (отчёт).
    hold = C.make_batch(128, random.Random(777_000), photos if len(photos) else None, args.p_photo, args.p_gray)
    val = C.read_split(args.val) if os.path.isdir(args.val) else None
    test_photos = C.PhotoBank([args.photo_test]) if os.path.isdir(args.photo_test) else []
    photo_hold = C.make_batch(32, random.Random(888_000), test_photos, 1.0, 0.0) if len(test_photos) else None

    model = C.UnrolledDOENet(args.depth, args.width, args.domain, refine=args.refine, refine_alpha=args.alpha).to(device)
    if args.init and not os.path.exists(os.path.join(args.run, "last.pt")):
        ini = torch.load(args.init, map_location=device, weights_only=False)["model"]
        own = model.state_dict()   # параметры другого размера (например, шаги уточнения) берём из новой модели
        ini = {k: v for k, v in ini.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(ini, strict=False)
        print(f"начальные веса: {args.init} (новые параметры: {missing})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print("параметров:", sum(p.numel() for p in model.parameters()), "| смешанная точность:", use_amp)

    start, best, elapsed0 = 0, -1.0, 0.0
    last_path, best_path = os.path.join(args.run, "last.pt"), os.path.join(args.run, "best.pt")
    CK_VERSION = 3   # версия постановки обучения; чекпоинты старых версий не продолжаем
    if os.path.exists(last_path) and torch.load(last_path, map_location="cpu", weights_only=False).get("version") != CK_VERSION:
        print("в папке чекпоинт старой версии обучения — начинаю заново (старые файлы переименованы в *.old)")
        for f in ("last.pt", "best.pt", "log.csv"):
            p_ = os.path.join(args.run, f)
            if os.path.exists(p_):
                os.replace(p_, p_ + ".old")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if "scaler" in ck and use_amp:
            scaler.load_state_dict(ck["scaler"])
        start, best, elapsed0 = ck["step"], ck["best"], ck.get("elapsed", 0.0)
        print(f"продолжаю с шага {start} ({elapsed0/3600:.2f} ч обучения), лучший контроль {best:.4f}")
    with open(os.path.join(args.run, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=1)

    if start == 0 and args.init:   # качество сразу после загрузки начальных весов (с новыми слоями уточнения)
        ev0 = "int" if args.domain == "int" else "amp"
        hp0, hs0 = evaluate(model, hold, device, domain=ev0)
        vp0, vs0 = evaluate(model, val, device, domain=ev0) if val is not None else (float("nan"),) * 2
        pp0, ps0 = evaluate(model, photo_hold, device, domain=ev0) if photo_hold is not None else (float("nan"),) * 2
        print(f"[     0] старт: контроль КК {hp0:.4f} ИСС {hs0:.4f} | val КК {vp0:.4f} ИСС {vs0:.4f} | фото КК {pp0:.4f} ИСС {ps0:.4f}", flush=True)

    budget = args.hours * 3600 if args.hours > 0 else None

    def progress(step, elapsed):
        """Доля пройденного обучения: по времени (если задано --hours) или по шагам."""
        return min(1.0, elapsed / budget) if budget else min(1.0, step / args.steps)

    def lr_at(step, elapsed):
        warm = min(1.0, step / 300)
        return args.lr * warm * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress(step, elapsed))))

    log_path = os.path.join(args.run, "log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["step", "loss", "train_P", "train_S", "hold_P", "hold_S", "val_P", "val_S", "photo_P", "photo_S", "lr", "hours"])

    rng = random.Random(args.seed * 1_000_003 + start)
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(C.make_batch, args.batch, rng, photos if len(photos) else None, args.p_photo, args.p_gray)
    t0 = time.time()
    run_loss, run_p, run_s, cnt = 0.0, 0.0, 0.0, 0
    mids = sorted({args.depth // 2 - 1, (3 * args.depth) // 4 - 1}) if args.depth >= 4 else []
    cfg = {"depth": args.depth, "width": args.width, "alpha": args.alpha, "domain": args.domain, "refine": args.refine}
    step = start
    gnorm = float("nan")
    while True:
        elapsed = elapsed0 + time.time() - t0
        finished = progress(step, elapsed) >= 1.0
        if finished and step == start:  # уже было обучено до конца
            print("обучение уже завершено ранее"); break
        if not finished:
            step += 1
            lr = lr_at(step, elapsed)
            for g in opt.param_groups:
                g["lr"] = lr
            codes = nxt.result()
            nxt = pool.submit(C.make_batch, args.batch, rng, photos if len(photos) else None, args.p_photo, args.p_gray)
            target = C.to_target(codes, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                phases = model(target, return_all=True)
            phases = [p_.float() for p_ in phases]
            loss, p, s = C.metric_loss(phases[-1], target, model.chirp, args.alpha, domain=args.domain)
            for m in mids:  # глубокое обучение промежуточных слоёв (стабилизирует развёртку)
                loss = loss + 0.1 * C.metric_loss(phases[m], target, model.chirp, args.alpha, domain=args.domain)[0]
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                scaler.step(opt)   # шаг пропускается автоматически при inf/NaN в градиентах
                scaler.update()
                run_loss += float(loss.detach()); run_p += float(p.mean()); run_s += float(s.mean()); cnt += 1
            if step % 50 == 0 or step == start + 5:
                el = time.time() - t0; done = step - start
                left = (budget - elapsed) if budget else el / done * (args.steps - step)
                print(f"  шаг {step}: {el/done:.2f} с/шаг, train ИСС {run_s/max(cnt,1):.4f} КК {run_p/max(cnt,1):.4f}, "
                      f"lr {lr:.1e}, |grad| {gnorm:.2e}, scale {scaler.get_scale():.0f}, до конца ~{max(left,0)/3600:.1f} ч", flush=True)

        if finished or step % args.eval_every == 0:
            ev = "int" if args.domain == "int" else "amp"
            hp, hs = evaluate(model, hold, device, domain=ev)
            vp, vs = evaluate(model, val, device, domain=ev) if val is not None else (float("nan"),) * 2
            pp, psv = evaluate(model, photo_hold, device, domain=ev) if photo_hold is not None else (float("nan"),) * 2
            if args.domain == "both":   # для страховочной модели выбираем по сумме обеих метрик
                ip, is_ = evaluate(model, hold, device, domain="int")
                print(f"         контроль по интенсивности: КК {ip:.4f} ИСС {is_:.4f}")
                score = (hp + hs + ip + is_) / 4
            else:   # ИСС — основная метрика организаторов: выбираем модель по взвешенной сумме
                score = args.alpha * hp + (1 - args.alpha) * hs
            print(f"[{step:6d}] train КК {run_p/max(cnt,1):.4f} ИСС {run_s/max(cnt,1):.4f} | "
                  f"контроль КК {hp:.4f} ИСС {hs:.4f} | val КК {vp:.4f} ИСС {vs:.4f} | фото КК {pp:.4f} ИСС {psv:.4f} | "
                  f"{elapsed/3600:.2f} ч", flush=True)
            log.writerow([step, run_loss / max(cnt, 1), run_p / max(cnt, 1), run_s / max(cnt, 1), hp, hs, vp, vs, pp, psv,
                          opt.param_groups[0]["lr"], elapsed / 3600]); logf.flush()
            run_loss, run_p, run_s, cnt = 0.0, 0.0, 0.0, 0
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": cfg, "step": step,
                            "hold": [hp, hs], "val": [vp, vs], "photo": [pp, psv]}, best_path)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "best": best, "elapsed": elapsed, "config": cfg, "version": CK_VERSION}, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)
        if finished:
            break
    print("готово. лучшие веса:", best_path)


if __name__ == "__main__":
    main()
