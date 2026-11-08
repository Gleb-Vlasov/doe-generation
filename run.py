"""Синтез фазового ДОЭ по комплектам из 4 изображений.

    python run.py --input <папка> --output <папка>

Вход: папка, в которой либо сразу лежат 4 PNG (один комплект), либо подпапки, по 4 PNG в каждой.
Порядок плоскостей 24.1 / 25.3 / 26.5 / 27.7 мм определяется по расстоянию в имени файла
(например «01_24.1mm.png»), а если его нет — по алфавитному порядку имён.

Выход для каждого комплекта <имя>:
    <output>/<имя>.png                ДОЭ: 128x128, 8 бит, оттенки серого; 0 -> -pi, 255 -> +pi
    <output>/<имя>/rec_*.png          4 восстановленных изображения (амплитуда, нормировка на максимум)
    <output>/<имя>/metrics.json       КК и ИСС по каждой плоскости, время работы сети
и общие <output>/results.csv, <output>/results.json (по комплектам и средние).
Восстановление и метрики считаются по ПРОЧИТАННОМУ с диска PNG ДОЭ (прямой БПФ, центрированная сетка).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from doe import core as C  # noqa: E402

IMG_EXT = (".png", ".PNG", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(IMG_EXT) and os.path.isfile(os.path.join(folder, f)))


def order_planes(names):
    """Сортирует 4 файла по расстоянию в имени (24.1 ... 27.7), иначе по имени."""
    def dist(n):
        m = re.findall(r"(2[4-7][.,]\d)", n)
        return float(m[-1].replace(",", ".")) if m else None
    d = [dist(n) for n in names]
    if all(x is not None for x in d) and len(set(d)) == len(d):
        return [n for _, n in sorted(zip(d, names))]
    return sorted(names)


def _group_flat(names):
    """Несколько комплектов в одной папке: группируем по общему началу имени (до расстояния),
    иначе — по 4 подряд в алфавитном порядке."""
    groups = {}
    for n in names:
        m = re.search(r"2[4-7][.,]\d", n)
        key = n[:m.start()].rstrip("_- .") if m else None
        groups.setdefault(key, []).append(n)
    if None not in groups and len(groups) > 1 and all(len(v) == 4 for v in groups.values()):
        return [(k or f"set_{i+1}", order_planes(v)) for i, (k, v) in enumerate(sorted(groups.items()))]
    # расстояние есть в каждом имени, а общего начала нет: k-й файл каждого расстояния -> k-й комплект
    by_d = {}
    for n in names:
        m = re.findall(r"(2[4-7][.,]\d)", n)
        by_d.setdefault(m[-1].replace(",", ".") if m else None, []).append(n)
    if None not in by_d and len(by_d) == 4 and len({len(v) for v in by_d.values()}) == 1:
        cols = [sorted(by_d[d]) for d in sorted(by_d, key=float)]
        return [(f"set_{k+1:02d}", [c[k] for c in cols]) for k in range(len(cols[0]))]
    names = sorted(names)
    return [(f"set_{i//4+1:02d}", names[i:i + 4]) for i in range(0, len(names) - len(names) % 4, 4)]


def find_sets(root):
    """Возвращает список (имя комплекта, папка, [4 файла] или None)."""
    imgs = list_images(root)
    if len(imgs) == 4 or (len(imgs) > 4 and len(imgs) % 4 != 0 and not any(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root))):
        return [(os.path.basename(os.path.normpath(root)) or "set", root, None)]
    if len(imgs) > 4:
        print(f"В папке {len(imgs)} изображений — считаю, что это {len(imgs)//4} комплектов по 4")
        return [(name, root, files) for name, files in _group_flat(imgs)]
    sets = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and len(list_images(p)) >= 4:
            sets.append((d, p, None))
    return sets


def read_set(folder, files=None):
    names = order_planes(files if files else list_images(folder))
    if len(names) != 4:
        print(f"  внимание: в {folder} {len(names)} изображений, беру первые 4 по порядку: {names[:4]}")
        names = names[:4]
    ims = []
    for n in names:
        with Image.open(os.path.join(folder, n)) as im:
            if im.mode in ("I", "F") or im.mode.startswith("I;16"):
                # 16/32-битные изображения: масштабируем по максимуму в 0..255 (иначе convert("L") обрежет значения)
                a = np.asarray(im, dtype=np.float64)
                a = a[..., 0] if a.ndim == 3 else a
                m = a.max()
                im = Image.fromarray(np.round(a / m * 255).astype(np.uint8) if m > 0 else np.zeros(a.shape, np.uint8))
            im = im.convert("L")
            if im.size != (C.N, C.N):
                print(f"  внимание: {n} размера {im.size}, привожу к 128x128")
                im = im.resize((C.N, C.N), Image.BILINEAR)
            ims.append(np.array(im))
    return np.stack(ims), names


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = C.UnrolledDOENet(cfg.get("depth", 24), cfg.get("width", 48), cfg.get("domain", "amp"),
                             refine=cfg.get("refine", 0), refine_alpha=cfg.get("alpha", 0.3))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description="Нейросетевой синтез фазового ДОЭ")
    ap.add_argument("--input", default="/data/input")
    ap.add_argument("--output", default="/data/output")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "model.pt"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-recon", action="store_true", help="не сохранять восстановленные изображения")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    if device.type == "cpu":
        # Тензоры 128x128 маленькие: больше 4 потоков только мешают (замер: 4 потока — 0.97 с, 16 потоков — 14 с)
        try:
            avail = len(os.sched_getaffinity(0))
        except AttributeError:
            avail = os.cpu_count() or 1
        torch.set_num_threads(max(1, min(4, avail)))
    model = load_model(args.weights, device)
    sets = find_sets(args.input)
    if not sets:
        sys.exit(f"Не найдено комплектов из 4 изображений в {args.input}")
    os.makedirs(args.output, exist_ok=True)
    print(f"Устройство: {device} | комплектов: {len(sets)}")

    with torch.no_grad():  # прогрев (первый вызов CUDA/cuDNN медленный и не отражает скорость сети)
        model(torch.zeros(1, 4, C.N, C.N, device=device) + 0.5)
        sync(device)

    rows = []
    for name, folder, files in sets:
        codes_in, names = read_set(folder, files)
        target = C.to_target(codes_in[None], device)
        with torch.no_grad():
            sync(device); t0 = time.perf_counter()
            phase = model(target)
            sync(device); dt = time.perf_counter() - t0
        doe_path = os.path.join(args.output, f"{name}.png")
        Image.fromarray(C.phase_to_codes(phase)[0].cpu().numpy(), mode="L").save(doe_path)

        # проверка по записанному файлу
        codes = torch.from_numpy(np.array(Image.open(doe_path))).to(device)
        with torch.no_grad():
            amp = C.propagate(C.codes_to_phase(codes)[None], model.chirp).abs()[0]
        rec8 = torch.round(C.norm_max(amp) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        p, s = C.scores(torch.from_numpy(rec8).float(), torch.from_numpy(codes_in).float())
        res = {"set": name, "inputs": names, "pearson": p.tolist(), "ssim": s.tolist(),
               "pearson_mean": float(p.mean()), "ssim_mean": float(s.mean()), "seconds": dt}
        if not args.no_recon:
            sub = os.path.join(args.output, name)
            os.makedirs(sub, exist_ok=True)
            for k, n in enumerate(names):
                Image.fromarray(rec8[k], mode="L").save(os.path.join(sub, "rec_" + os.path.splitext(n)[0] + ".png"))
            with open(os.path.join(sub, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
        rows.append(res)
        print(f"{name:20s} ИСС {res['ssim_mean']:.4f}  КК {res['pearson_mean']:.4f}  время {dt*1000:.1f} мс")

    summary = {"device": str(device), "sets": rows,
               "ssim_mean": float(np.mean([r["ssim_mean"] for r in rows])),
               "pearson_mean": float(np.mean([r["pearson_mean"] for r in rows])),
               "seconds_mean": float(np.mean([r["seconds"] for r in rows]))}
    with open(os.path.join(args.output, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.output, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "ssim_mean", "pearson_mean", "seconds"] + [f"ssim_{i+1}" for i in range(4)] + [f"pearson_{i+1}" for i in range(4)])
        for r in rows:
            w.writerow([r["set"], r["ssim_mean"], r["pearson_mean"], r["seconds"]] + r["ssim"] + r["pearson"])
    print(f"ИТОГО ({len(rows)} комплектов): ИСС {summary['ssim_mean']:.4f}  КК {summary['pearson_mean']:.4f}  "
          f"среднее время {summary['seconds_mean']*1000:.1f} мс")


if __name__ == "__main__":
    main()
