"""Batch shop-type typing from real storefront images, via CLIP zero-shot.

For each outlet image: score it against text prompts for each shop-type PLUS
"unusable" classes (selfie/person, blurry/wall, indoor-event). If the top class
is unusable or low-confidence, the image is rejected (usable=False) and typing
falls back to text. This is the usability gate the field photos demand.

Writes data/image_types.parquet: outletid, image_format, image_conf, usable.
"""
from __future__ import annotations

import glob
import os

import open_clip
import polars as pl
import torch
from PIL import Image

MODEL, PRE = "ViT-B-16-SigLIP", "webli"   # SigLIP beats openai CLIP on this task
IMG_DIR = "data/imgs"
BATCH = 64
MIN_CONF = 0.0    # SigLIP is sigmoid-scored; rely on the distractor classes, not a threshold
TEMPLATES = ["a photo of {}", "a storefront photo of {}", "the front of {}", "{}"]

# discriminative prompts (V2, validated in tune_clip.py: kirana 0.88 / chemist 0.81)
CLASSES = {
    "kirana": ["a neighbourhood grocery and provision shop with shelves packed with biscuits, snacks, soaps, spices and daily household packets behind a counter"],
    "chemist": ["a pharmacy medical store with a counter and shelves of medicine boxes, tablets, syrups and health products"],
    "supermarket": ["a large self-service supermarket with wide aisles, shopping trolleys and long rows of shelves"],
    "pan_kiosk": ["a tiny cigarette and paan kiosk that is mostly hanging tobacco and gutka sachets at a small window, with almost no grocery shelves"],
    "cosmetics": ["a cosmetics and beauty shop displaying lipstick, makeup, perfume, shampoo and skincare bottles"],
    "horeca": ["a place serving prepared food to eat: a restaurant, cafe, tea stall, sweet shop or bakery with a food counter"],
    "wholesale": ["a wholesale godown with stacked cartons, gunny sacks and bulk cases of stock"],
    "_bad_person": ["a close-up selfie or portrait of a person's face", "a group of people posing at a meeting or event"],
    "_bad_blur": ["a blurry out-of-focus photo, or a photo of a plain wall, floor or ceiling, or a dark image"],
}


def _load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRE)
    model.eval()
    tok = open_clip.get_tokenizer(MODEL)
    # one averaged text embedding per label (over its prompts x templates)
    labels, vecs = [], []
    with torch.no_grad():
        for lab, ps in CLASSES.items():
            texts = [t.format(p) for p in ps for t in TEMPLATES]
            e = model.encode_text(tok(texts))
            e = e / e.norm(dim=-1, keepdim=True)
            v = e.mean(dim=0)
            vecs.append(v / v.norm())
            labels.append(lab)
        txt = torch.stack(vecs)
    return model, preprocess, labels, txt


def classify(model, preprocess, labels, txt, paths):
    imgs, ok = [], []
    for p in paths:
        try:
            imgs.append(preprocess(Image.open(p).convert("RGB"))); ok.append(p)
        except Exception:
            pass
    if not imgs:
        return []
    with torch.no_grad():
        im = model.encode_image(torch.stack(imgs))
        im = im / im.norm(dim=-1, keepdim=True)
        probs = (100.0 * im @ txt.T).softmax(dim=-1)   # [n, n_labels]
    out = []
    for r, p in enumerate(ok):
        j = int(probs[r].argmax())
        top, conf = labels[j], float(probs[r, j])
        usable = (not top.startswith("_bad")) and conf >= MIN_CONF
        oid = int(os.path.splitext(os.path.basename(p))[0])
        out.append((oid, top if usable else None, round(conf, 3), usable))
    return out


def main() -> None:
    paths = sorted(glob.glob(f"{IMG_DIR}/*.jpg"))
    print(f"images on disk: {len(paths)}", flush=True)
    model, preprocess, labels, txt = _load_model()
    rows = []
    for i in range(0, len(paths), BATCH):
        try:
            rows.extend(classify(model, preprocess, labels, txt, paths[i:i + BATCH]))
        except Exception as e:                       # one corrupt image can't kill the run
            for p in paths[i:i + BATCH]:             # fall back to one-at-a-time
                try:
                    rows.extend(classify(model, preprocess, labels, txt, [p]))
                except Exception:
                    pass
        if (i // BATCH) % 10 == 0:
            print(f"  {i+BATCH}/{len(paths)}", flush=True)
    df = pl.DataFrame(rows, schema=["outletid", "image_format", "image_conf", "usable"], orient="row")
    df.write_parquet("data/image_types.parquet")
    usable = df.filter(pl.col("usable"))
    print(f"\nclassified {df.height} | usable {usable.height} ({round(100*usable.height/max(df.height,1))}%)")
    print(usable.group_by("image_format").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
