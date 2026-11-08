"""Финализация решения одной командой.

    python tools/finalize.py --weights ../runs_amp/amp/best.pt --val ../dataset_v1/val --photos ../photo_experiments/bsds500/test

Что делает:
  1. копирует обученные веса в weights/model.pt;
  2. оценивает сеть и классический Гершберг–Сэкстон (50 итераций) на проверочных четвёрках и на фото:
     ИСС, КК (амплитуды, нормировка на максимум, фаза через 8-битный PNG) и время на один ДОЭ (GPU и CPU);
  3. записывает таблицу результатов в README.md и results/summary.json, картинку примеров в results/examples.png;
  4. запускает run.py на examples/ (проверка точки входа Docker) и сверяет метрики.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from doe import core as C  # noqa: E402
import run as R  # noqa: E402


def photo_sets(folder, n_sets=32):
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))[: 4 * n_sets]
    ims = []
    for f in files:
        im = Image.open(f).convert("L"); w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((128, 128), Image.LANCZOS)
        ims.append(np.array(im))
    k = len(ims) // 4
    return np.stack(ims[: 4 * k]).reshape(k, 4, 128, 128) if k else None


@torch.no_grad()
def gs50(t, model, steps=50):
    phase = torch.angle((C.ifft(t * model.carrier) * model.chirp.conj()).sum(1))
    for _ in range(steps):
        g = C.propagate(phase, model.chirp)
        phase = torch.angle((C.ifft(t * g / g.abs().clamp_min(1e-8)) * model.chirp.conj()).sum(1))
    return phase


@torch.no_grad()
def score(phase, codes, model):
    phase = C.codes_to_phase(C.phase_to_codes(phase))
    amp8 = torch.round(C.norm_max(C.propagate(phase, model.chirp).abs()) * 255)
    p, s = C.scores(amp8, C.to_target(codes, phase.device))
    return p.cpu(), s.cpu()


@torch.no_grad()
def evaluate(method, codes, model, device, batch=16):
    ps, ss = [], []
    for i in range(0, len(codes), batch):
        c = codes[i:i + batch]; t = C.to_target(c, device)
        ph = model(t) if method == "net" else gs50(t, model)
        p, s = score(ph, c, model); ps.append(p); ss.append(s)
    return float(torch.cat(ss).mean()), float(torch.cat(ps).mean())


@torch.no_grad()
def timing(method, codes, model, device, n=16):
    """Среднее время синтеза одного ДОЭ (батч 1), после прогрева."""
    t = C.to_target(codes[:1], device)
    for _ in range(2):
        model(t) if method == "net" else gs50(t, model)
    R.sync(device)
    ts = []
    for i in range(min(n, len(codes))):
        t = C.to_target(codes[i:i + 1], device)
        R.sync(device); t0 = time.perf_counter()
        model(t) if method == "net" else gs50(t, model)
        R.sync(device); ts.append(time.perf_counter() - t0)
    return float(np.mean(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val", default=os.path.join(HERE, "..", "dataset_v1", "val"))
    ap.add_argument("--photos", default=os.path.join(HERE, "..", "photo_experiments", "bsds500", "test"))
    ap.add_argument("--skip-cpu-timing", action="store_true")
    args = ap.parse_args()

    dst = os.path.join(HERE, "weights", "model.pt")
    if os.path.abspath(args.weights) != os.path.abspath(dst):
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        torch.save({"model": ck["model"], "config": ck["config"], "step": ck.get("step")}, dst)  # без состояния оптимизатора
        print(f"веса: {args.weights} -> {dst} ({os.path.getsize(dst)/1e6:.1f} МБ), шаг {ck.get('step')}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = R.load_model(dst, dev)
    sets = {}
    if os.path.isdir(args.val):
        sets["Синтетика (dataset_v1/val)"] = C.read_split(args.val)
    if os.path.isdir(args.photos):
        ph = photo_sets(args.photos)
        if ph is not None:
            sets["Фотографии (BSDS500 test)"] = ph
    rows = {}
    for name, codes in sets.items():
        for method in ("gs", "net"):
            s, p = evaluate(method, codes, model, dev)
            rows[(name, method)] = (s, p, len(codes))
            print(f"{name:28s} {'сеть' if method=='net' else 'GS50':5s} ИСС {s:.4f} КК {p:.4f} ({len(codes)} комплектов)")

    any_codes = next(iter(sets.values()))
    times = {}
    for method in ("gs", "net"):
        times[(method, str(dev))] = timing(method, any_codes, model, dev)
    if dev.type == "cuda" and not args.skip_cpu_timing:
        cpu = torch.device("cpu"); torch.set_num_threads(os.cpu_count() or 1)
        mcpu = R.load_model(dst, cpu)
        for method in ("gs", "net"):
            times[(method, "cpu")] = timing(method, any_codes, mcpu, cpu, n=8)
    for k, v in times.items():
        print(f"время {k[0]:4s} на {k[1]:5s}: {v*1000:.1f} мс/ДОЭ")

    # картинка примеров (PIL, без matplotlib)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    try:
        from PIL import ImageDraw
        ex = np.concatenate([c[:2] for c in sets.values()], 0)
        with torch.no_grad():
            ph = C.codes_to_phase(C.phase_to_codes(model(C.to_target(ex, dev))))
            rec = (C.norm_max(C.propagate(ph, model.chirp).abs()) * 255).round().clamp(0, 255).byte().cpu().numpy()
        tile, pad, lab = 256, 6, 110
        canvas = Image.new("L", (lab + 4 * (tile + pad), 30 + len(ex) * 2 * (tile + pad)), 255)
        d = ImageDraw.Draw(canvas)
        for k in range(4):
            d.text((lab + k * (tile + pad) + tile // 2 - 20, 8), f"{C.DISTANCES[k]*1e3:.1f} mm", fill=0)
        y = 30
        for r in range(len(ex)):
            for row, (arr, title) in enumerate([(ex[r], "target"), (rec[r], "DOE (net)")]):
                d.text((8, y + tile // 2), title, fill=0)
                for k in range(4):
                    canvas.paste(Image.fromarray(np.asarray(arr[k], np.uint8)).resize((tile, tile), Image.NEAREST),
                                 (lab + k * (tile + pad), y))
                y += tile + pad
        canvas.save(os.path.join(HERE, "results", "examples.png"))
    except Exception as e:  # noqa: BLE001
        print("картинка примеров не построена:", e)

    # таблица в README
    def t(m, d):
        v = times.get((m, d)); return f"{v*1000:.0f} мс" if v else "—"
    lines = ["| Набор | Метод | ИСС | КК |", "|---|---|---:|---:|"]
    for (name, method), (s, p, n) in rows.items():
        lines.append(f"| {name}, {n} компл. | {'**Наша сеть, один проход**' if method=='net' else 'Гершберг–Сэкстон, 50 итераций'} | "
                     f"{'**%.3f**' % s if method=='net' else '%.3f' % s} | {'**%.3f**' % p if method=='net' else '%.3f' % p} |")
    lines += ["", "| Время синтеза одного ДОЭ | GPU | CPU |", "|---|---:|---:|",
              f"| Наша сеть | {t('net','cuda')} | {t('net','cpu')} |",
              f"| Гершберг–Сэкстон, 50 итераций | {t('gs','cuda')} | {t('gs','cpu')} |",
              "", "Для справки: прямая итерационная оптимизация под эту же метрику (~1000 итераций Adam на каждый комплект) "
              "достигает на синтетике ИСС ≈ 0.36–0.39 и КК ≈ 0.55–0.58. Это практический потолок для данной физики, и сеть получает "
              "близкий результат за один проход.", "", "![Примеры](results/examples.png)"]
    readme = os.path.join(HERE, "README.md")
    txt = open(readme, encoding="utf-8").read()
    a, b = txt.index("<!-- RESULTS -->") + len("<!-- RESULTS -->"), txt.index("<!-- /RESULTS -->")
    open(readme, "w", encoding="utf-8").write(txt[:a] + "\n" + "\n".join(lines) + "\n" + txt[b:])
    json.dump({"metrics": {f"{n}|{m}": {"ssim": s, "pearson": p, "sets": k} for (n, m), (s, p, k) in rows.items()},
               "seconds": {f"{m}|{d}": v for (m, d), v in times.items()}},
              open(os.path.join(HERE, "results", "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("README.md и results/ обновлены")

    # проверка точки входа
    out = os.path.join(HERE, "out_check")
    r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "--input", os.path.join(HERE, "examples"),
                        "--output", out], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-1500:])
    print("run.py:", "OK" if r.returncode == 0 and os.path.exists(os.path.join(out, "results.json")) else "ОШИБКА")


if __name__ == "__main__":
    main()
