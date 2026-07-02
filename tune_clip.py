"""Iterate CLIP prompt strategies for shop-typing, measured on an auto-eval.

Eval labels = system shoptype mapped to canonical class (reliable for the
distinctive classes; kirana/general is the broad default). We optimise prompt
sets + negative-prompting tricks (mean-centering, explicit distractor classes)
to raise per-class recall — especially to stop kiranas being called pan/horeca.
"""
from __future__ import annotations

import glob
import os

import open_clip
import polars as pl
import torch
from PIL import Image

from engine.segment import _clean_format

MODEL, PRE = "ViT-B-32-quickgelu", "openai"
TEMPLATES = ["a photo of {}", "a storefront photo of {}", "the front of {}", "{}"]
TARGET = {"kirana", "chemist", "supermarket", "pan_kiosk", "cosmetics", "horeca", "wholesale"}
PER_CLASS = 100

# ── prompt strategies ─────────────────────────────────────────────────────
V1 = {  # baseline (what shipped)
    "kirana": ["a small Indian kirana grocery store with packaged goods"],
    "chemist": ["a pharmacy or chemist medical store with medicines"],
    "supermarket": ["a modern supermarket or departmental store with aisles"],
    "pan_kiosk": ["a small paan, cigarette and snacks kiosk or roadside stall"],
    "cosmetics": ["a cosmetics and beauty products shop"],
    "horeca": ["a restaurant, hotel, cafe, bakery or food-service outlet"],
    "wholesale": ["a wholesale distributor godown with bulk cartons and sacks"],
}
V2 = {  # discriminative — describe the visually distinctive features
    "kirana": ["a neighbourhood grocery and provision shop with shelves packed with biscuits, snacks, soaps, spices and daily household packets behind a counter"],
    "chemist": ["a pharmacy medical store with a counter and shelves of medicine boxes, tablets, syrups and health products"],
    "supermarket": ["a large self-service supermarket with wide aisles, shopping trolleys and long rows of shelves"],
    "pan_kiosk": ["a tiny cigarette and paan kiosk that is mostly hanging tobacco and gutka sachets at a small window, with almost no grocery shelves"],
    "cosmetics": ["a cosmetics and beauty shop displaying lipstick, makeup, perfume, shampoo and skincare bottles"],
    "horeca": ["a place serving prepared food to eat: a restaurant, cafe, tea stall, sweet shop or bakery with a food counter"],
    "wholesale": ["a wholesale godown with stacked cartons, gunny sacks and bulk cases of stock"],
}
# distractor classes that absorb non-shop / ambiguous photos (negative prompting)
DISTRACT = {
    "_person": ["a close-up photo or selfie of a person's face", "people posing together"],
    "_scene": ["an empty street, a road, houses or a parked vehicle", "an indoor room, office or event hall"],
    "_blur": ["a blurry dark out-of-focus photo, or a photo of a plain wall, floor or ceiling"],
}


def load_model():
    m, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRE)
    m.eval()
    return m, pre, open_clip.get_tokenizer(MODEL)


def label_vecs(model, tok, classes):
    labels, vs = [], []
    with torch.no_grad():
        for lab, ps in classes.items():
            texts = [t.format(p) for p in ps for t in TEMPLATES]
            e = model.encode_text(tok(texts)); e = e / e.norm(dim=-1, keepdim=True)
            v = e.mean(0); vs.append(v / v.norm()); labels.append(lab)
    return labels, torch.stack(vs)


def run_config(name, classes, model, pre, tok, evalset, center=False, distract=False):
    cls = dict(classes)
    if distract:
        cls.update(DISTRACT)
    labels, txt = label_vecs(model, tok, cls)
    if center:                                   # negative prompting: remove the common direction
        shop_idx = [i for i, l in enumerate(labels) if not l.startswith("_")]
        mean = txt[shop_idx].mean(0, keepdim=True)
        txt = txt - mean
        txt = txt / txt.norm(dim=-1, keepdim=True)
    preds = {}
    B = 64
    paths = list(evalset.keys())
    for i in range(0, len(paths), B):
        chunk = paths[i:i + B]
        imgs, ok = [], []
        for p in chunk:
            try:
                imgs.append(pre(Image.open(p).convert("RGB"))); ok.append(p)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            im = model.encode_image(torch.stack(imgs)); im = im / im.norm(dim=-1, keepdim=True)
            sims = im @ txt.T
        for r, p in enumerate(ok):
            j = int(sims[r].argmax())
            preds[p] = labels[j]
    # score
    per = {c: [0, 0] for c in TARGET}   # [correct, total]
    kir_conf = {}
    for p, truth in evalset.items():
        pred = preds.get(p)
        if pred is None:
            continue
        per[truth][1] += 1
        if pred == truth:
            per[truth][0] += 1
        if truth == "kirana" and pred != "kirana":
            kir_conf[pred] = kir_conf.get(pred, 0) + 1
    tot_c = sum(v[0] for v in per.values()); tot_n = sum(v[1] for v in per.values())
    acc = round(tot_c / tot_n, 3) if tot_n else 0
    recalls = {c: round(v[0] / v[1], 2) for c, v in per.items() if v[1] >= 5}
    print(f"\n[{name}] overall acc={acc}  (center={center} distract={distract})")
    print("   recall:", recalls)
    print("   kirana mis-called as:", dict(sorted(kir_conf.items(), key=lambda x: -x[1])))
    return acc


def main():
    d = pl.read_parquet("data/outlets_geo2.parquet").filter(pl.col("has_data"))
    have = {int(os.path.basename(p)[:-4]) for p in glob.glob("data/imgs/*.jpg")}
    d = d.filter(pl.col("outletid").is_in(list(have)))
    d = d.with_columns(canon=pl.col("shoptypename").map_elements(_clean_format, return_dtype=pl.Utf8))
    d = d.filter(pl.col("canon").is_in(list(TARGET)) & pl.col("shoptypename").is_not_null()
                 & (pl.col("shoptypename") != ""))
    ev = d.group_by("canon").head(PER_CLASS)
    evalset = {f"data/imgs/{r['outletid']}.jpg": r["canon"]
               for r in ev.select("outletid", "canon").iter_rows(named=True)}
    print(f"eval images: {len(evalset)}")
    print("eval label mix:", dict(zip(*ev.group_by("canon").len().sort("canon").to_dict(as_series=False).values())))
    model, pre, tok = load_model()
    run_config("V1 baseline", V1, model, pre, tok, evalset)
    run_config("V2 discriminative", V2, model, pre, tok, evalset)
    run_config("V2 + centering", V2, model, pre, tok, evalset, center=True)
    run_config("V2 + centering + distractors", V2, model, pre, tok, evalset, center=True, distract=True)


if __name__ == "__main__":
    main()
