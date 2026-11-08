"""Ядро: физика Френеля, метрика организаторов (амплитуды, нормировка на максимум), сеть, данные.

Правила оценки (уточнение организаторов):
  1) сравниваются амплитуды;
  2) каждое изображение перед сравнением делится на свой максимум.
Метрики: Пирсон (КК) и глобальный ИСС (формула со слайда, c1=0.01^2, c2=0.03^2).
"""
from __future__ import annotations

import glob
import math
import os
import random
import string
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image, ImageDraw, ImageFont
from torch import nn

N = 128
WAVELENGTH = 532e-9
PITCH = 10e-6
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
FILENAMES = ("01_24.1mm.png", "02_25.3mm.png", "03_26.5mm.png", "04_27.7mm.png")
AX = (-2, -1)


# ------------------------------------------------------------------ physics
def make_chirp():
    axis = (torch.arange(N, dtype=torch.float64) - N // 2) * PITCH
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    z = torch.tensor(DISTANCES, dtype=torch.float64)[:, None, None]
    return torch.exp(1j * math.pi * (x**2 + y**2) / (WAVELENGTH * z)).to(torch.complex64)


def fft(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=AX), dim=AX, norm="ortho"), dim=AX)


def ifft(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=AX), dim=AX, norm="ortho"), dim=AX)


def propagate(phase, chirp):
    """phase [B,N,N] -> complex field [B,4,N,N]. Тот же расчёт, что simulator.py (масштаб не важен после нормировки)."""
    return fft(torch.exp(1j * phase)[:, None] * chirp)


# ------------------------------------------------------------------ phase PNG
def phase_to_codes(phase):
    return ((phase + math.pi).remainder(2 * math.pi) * (255 / (2 * math.pi))).round().clamp(0, 255).to(torch.uint8)


def codes_to_phase(codes):
    return codes.float() * (2 * math.pi / 255) - math.pi


def png_phase_ste(phase):
    q = codes_to_phase(phase_to_codes(phase.detach()))
    return phase + (q - phase).detach()


# ------------------------------------------------------------------ metrics
def norm_max(x):
    return x / x.amax(AX, keepdim=True).clamp_min(1e-12)


def scores(rec_amp, target_amp):
    """[...,N,N] -> (pearson, ssim) [...]. Оба изображения нормируются на максимум."""
    a, b = norm_max(rec_amp), norm_max(target_amp)
    ma, mb = a.mean(AX, True), b.mean(AX, True)
    ac, bc = a - ma, b - mb
    va, vb = ac.square().mean(AX, True), bc.square().mean(AX, True)
    cov = (ac * bc).mean(AX, True)
    pearson = cov / (va.clamp_min(1e-12).sqrt() * vb.clamp_min(1e-12).sqrt())
    ssim = ((2 * ma * mb + 1e-4) * (2 * cov + 9e-4)) / ((ma**2 + mb**2 + 1e-4) * (va + vb + 9e-4))
    return pearson[..., 0, 0], ssim[..., 0, 0]


def recon(field, domain):
    """Что сравнивается с целью: 'amp' -> |поле|, 'int' -> |поле|^2."""
    a = field.abs()
    return a.square() if domain == "int" else a


def metric_loss(phase, target, chirp, alpha=0.4, quantize=True, domain="amp"):
    """1 - (alpha*КК + (1-alpha)*ИСС) после PNG-квантования фазы.
    domain: 'amp' (амплитуды), 'int' (интенсивности) или 'both' (страховка: среднее двух)."""
    if quantize:
        phase = png_phase_ste(phase)
    field = propagate(phase, chirp)
    if domain == "both":
        pa, sa = scores(recon(field, "amp"), target)
        pi, si = scores(recon(field, "int"), target)
        loss = 1 - 0.5 * ((alpha * pa + (1 - alpha) * sa).mean() + (alpha * pi + (1 - alpha) * si).mean())
        return loss, pa.detach(), sa.detach()
    p, s = scores(recon(field, domain), target)
    return 1 - (alpha * p + (1 - alpha) * s).mean(), p.detach(), s.detach()


# ------------------------------------------------------------------ safe ops
class _SafeAngle(torch.autograd.Function):
    """angle(z) с ограниченной производной: |z|^2 в знаменателе не меньше eps^2.
    Прямой проход точно равен torch.angle; без ограничения производная 1/|z| взрывается в нулях поля."""

    @staticmethod
    def forward(ctx, z, eps):
        ctx.save_for_backward(z, eps)
        return torch.angle(z)

    @staticmethod
    def backward(ctx, grad):
        z, eps = ctx.saved_tensors
        denom = (z.real.square() + z.imag.square()).clamp_min(eps.square())
        return grad * 1j * z / denom, None


def safe_angle(z, rel=0.05):
    eps = (rel * z.abs().square().mean(AX, True).sqrt()).detach().clamp_min(1e-6)
    return _SafeAngle.apply(z, eps)


# ------------------------------------------------------------------ network
class Stage(nn.Module):
    """Обучаемая поправка одного физического слоя: совместно по 4 плоскостям."""

    def __init__(self, width=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(20, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 8, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class UnrolledDOENet(nn.Module):
    """Развёрнутая физически-информированная сеть (deep unfolding).

    Каждый из K слоёв: Френель вперёд (4 плоскости) -> CNN по признакам всех плоскостей
    предсказывает желаемую амплитуду и фазовый сдвиг в каждой плоскости -> обратный Френель
    -> обновление фазы ДОЭ с обучаемыми шагом и инерцией. Веса слоёв раздельные.
    На этапе применения — один прямой проход, без оптимизатора.
    """

    def __init__(self, depth=24, width=48, domain="amp", refine=0, refine_alpha=0.3):
        super().__init__()
        self.depth = depth
        self.domain = domain
        self.refine = refine
        self.refine_alpha = refine_alpha
        # слои уточнения по градиенту метрики (развёрнутый Adam с обучаемым шагом каждого слоя)
        self.refine_log_lr = nn.Parameter(torch.full((max(refine, 1),), math.log(0.05)))
        self.register_buffer("chirp", make_chirp(), persistent=False)
        g = torch.Generator().manual_seed(42)
        self.register_buffer("carrier", torch.exp(2j * math.pi * torch.rand(4, N, N, generator=g)), persistent=False)
        self.stages = nn.ModuleList([Stage(width) for _ in range(depth)])
        self.log_relax = nn.Parameter(torch.zeros(depth))
        self.momentum = nn.Parameter(torch.zeros(depth))
        self.plane_logits = nn.Parameter(torch.zeros(depth, 4))

    def _layer(self, k, phase, velocity, target, tt):
        """Один физический слой: Френель вперёд -> CNN-поправки -> Френель назад -> обновление фазы."""
        chirp = self.chirp
        g = fft(torch.exp(1j * phase)[:, None] * chirp)
        rms = g.abs().square().mean(AX, True).sqrt()
        a = g.abs().clamp_min(1e-8)
        s = (a * target).sum(AX, True) / tt                    # МНК-масштаб цели
        unit = g / torch.maximum(a, 0.05 * rms.detach())      # фазовый множитель поля, без взрыва в нулях
        peak = a.amax(AX, True)
        feats = torch.cat([target, a / peak, a / rms, (s * target - a) / rms, target - a / peak], 1)
        out = self.stages[k](feats).float()
        desired = (s * target + rms * torch.tanh(out[:, :4])).clamp_min(0)
        rot = 0.5 * math.pi * torch.tanh(out[:, 4:])
        upd = desired * unit * torch.exp(1j * rot)
        w = torch.softmax(self.plane_logits[k], 0)[None, :, None, None] * 4
        back = (ifft(upd) * chirp.conj() * w).sum(1)
        delta = safe_angle(back * torch.exp(-1j * phase))
        velocity = self.momentum[k].clamp(-0.5, 0.95) * velocity + delta
        phase = phase + self.log_relax[k].exp().clamp(max=2.0) * velocity
        return phase, velocity

    def metric_grad(self, phase, target):
        """Градиент метрики (alpha*КК + (1-alpha)*ИСС по амплитудам) по фазе ДОЭ — вход слоя уточнения.
        Считается как отдельный вход (без производных второго порядка)."""
        with torch.enable_grad():
            p = phase.detach().requires_grad_(True)
            pk, sk = scores(propagate(p, self.chirp).abs(), target.detach())
            loss = -(self.refine_alpha * pk + (1 - self.refine_alpha) * sk).sum()
            g, = torch.autograd.grad(loss, p)
        return g.detach()

    def forward(self, target, return_all=False, checkpoint=None):
        """target: амплитуды [B,4,N,N], нормированные на максимум. Возвращает фазу [B,N,N] (радианы).
        checkpoint: пересчитывать слои при обратном проходе вместо хранения (экономит память GPU при обучении)."""
        if checkpoint is None:
            checkpoint = self.training and torch.is_grad_enabled()
        chirp = self.chirp
        if self.domain == "int":          # цель — интенсивность: физические слои целятся в амплитуду sqrt(I)
            target = target.clamp_min(0).sqrt()
        phase = torch.angle((ifft(target * self.carrier) * chirp.conj()).sum(1))
        velocity = torch.zeros_like(phase)
        tt = target.square().sum(AX, True).clamp_min(1e-6)
        outs = []
        for k in range(self.depth):
            if checkpoint:
                phase, velocity = torch.utils.checkpoint.checkpoint(
                    self._layer, k, phase, velocity, target, tt, use_reentrant=False)
            else:
                phase, velocity = self._layer(k, phase, velocity, target, tt)
            if return_all:
                outs.append(phase)
        if self.refine > 0:
            m = torch.zeros_like(phase); v = torch.zeros_like(phase)
            b1, b2 = 0.9, 0.999
            for t in range(1, self.refine + 1):
                g = self.metric_grad(phase, target)
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                step = (m / (1 - b1**t)) / ((v / (1 - b2**t)).sqrt() + 1e-8)
                phase = phase - self.refine_log_lr[t - 1].exp() * step
            if return_all:
                outs.append(phase)
        return outs if return_all else phase


# ------------------------------------------------------------------ data
FONT_DIRS = ["C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts"]
FONT_NAMES = ["arial.ttf", "arialbd.ttf", "consola.ttf", "times.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]


def _font_paths():
    out = []
    for d in FONT_DIRS:
        for n in FONT_NAMES:
            p = os.path.join(d, n)
            if os.path.exists(p):
                out.append(p)
    return out


FONT_PATHS = _font_paths()


@lru_cache(maxsize=256)
def get_font(path, size):
    if path is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(path, size=size)


def draw_shape_image(rng):
    """Тот же генератор, что в doe_train.py, + кольца (как на слайде организаторов)."""
    image = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(image)
    kind = rng.choice(["symbol", "symbol", "shapes", "shapes", "lines", "rings"])
    if kind == "symbol":
        alphabet = string.ascii_uppercase + string.digits
        text = "".join(rng.choices(alphabet, k=rng.choice([1, 1, 2])))
        font = get_font(rng.choice(FONT_PATHS) if FONT_PATHS else None, rng.randint(26, 60))
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        x = rng.randint(0, max(0, N - (r - l))) - l
        y = rng.randint(0, max(0, N - (b - t))) - t
        draw.text((x, y), text, font=font, fill=255)
    elif kind == "shapes":
        for _ in range(rng.randint(1, 4)):
            w, h = rng.randint(12, 65), rng.randint(12, 65)
            x, y = rng.randrange(N - w), rng.randrange(N - h)
            fill = rng.choice([None, 255])
            if rng.random() < 0.5:
                draw.ellipse((x, y, x + w, y + h), fill=fill, outline=255, width=rng.randint(2, 6))
            else:
                draw.rectangle((x, y, x + w, y + h), fill=fill, outline=255, width=rng.randint(2, 6))
    elif kind == "rings":
        for _ in range(rng.randint(1, 3)):
            r = rng.randint(8, 50)
            cx, cy = rng.randint(r, N - r - 1), rng.randint(r, N - r - 1)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=255, width=rng.randint(2, 6))
    else:
        for _ in range(rng.randint(1, 3)):
            pts = [(rng.randrange(N), rng.randrange(N)) for _ in range(rng.randint(2, 5))]
            draw.line(pts, fill=255, width=rng.randint(2, 7))
    return (np.asarray(image) > 127).astype(np.float32)


class PhotoBank:
    """Случайные фрагменты фотографий (BSDS500) в оттенках серого."""

    def __init__(self, folders):
        files = []
        for f in folders:
            files += sorted(glob.glob(os.path.join(f, "*.jpg"))) + sorted(glob.glob(os.path.join(f, "*.png")))
        self.images = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255 for p in files]

    def __len__(self):
        return len(self.images)

    def sample(self, rng):
        im = self.images[rng.randrange(len(self.images))]
        h, w = im.shape
        s = rng.randint(min(96, min(h, w)), min(h, w))
        y, x = rng.randint(0, h - s), rng.randint(0, w - s)
        crop = Image.fromarray((im[y:y + s, x:x + s] * 255).astype(np.uint8)).resize((N, N), Image.BILINEAR)
        out = np.asarray(crop, dtype=np.float32) / 255
        if rng.random() < 0.5:
            out = out[:, ::-1]
        return np.ascontiguousarray(out)


def gray_version(img, rng):
    """Полутоновая версия бинарной картинки (текстура, размытие, фон)."""
    yy, xx = np.mgrid[0:N, 0:N] / N
    tex = 0.5 + 0.5 * np.sin(2 * np.pi * (rng.random() * 3 * xx + rng.random() * 3 * yy) + rng.random() * 6)
    k = rng.choice([1, 3, 5, 9])
    soft = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((N // k, N // k), Image.BILINEAR).resize((N, N), Image.BILINEAR), dtype=np.float32) / 255 if k > 1 else img
    out = (0.3 + 0.7 * rng.random()) * img * (0.4 + 0.6 * tex) + rng.random() * 0.6 * soft + rng.random() * 0.15
    return out


def make_batch(batch, rng, photos=None, p_photo=0.25, p_gray=0.2):
    """Возвращает uint8-цели [B,4,N,N] (как прочитанные PNG)."""
    out = np.zeros((batch, 4, N, N), np.float32)
    for i in range(batch):
        r = rng.random()
        for k in range(4):
            if photos is not None and len(photos) and r < p_photo:
                im = photos.sample(rng)
            else:
                im = draw_shape_image(rng)
                if r > 1 - p_gray:
                    im = gray_version(im, rng)
            out[i, k] = im
    out = out / np.maximum(out.max((-2, -1), keepdims=True), 1e-6)
    return np.round(out * 255).astype(np.uint8)


def to_target(codes_uint8, device):
    t = torch.as_tensor(codes_uint8, device=device).float()
    return norm_max(t)


def read_quartet(folder):
    ims = []
    for name in FILENAMES:
        p = os.path.join(folder, name)
        with Image.open(p) as im:
            a = np.array(im.convert("L"))
        if a.shape != (N, N):
            raise ValueError(f"{p}: ожидается 128x128, получено {a.shape}")
        ims.append(a)
    return np.stack(ims)


def read_split(root, limit=None):
    folders = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))[:limit]
    return np.stack([read_quartet(f) for f in folders])
