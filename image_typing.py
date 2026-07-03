"""Storefront-image typing with a progress callback.

Reuses clip_classify's validated image model. Resolves an outlet's photo from a
local cache (data/imgs/{outletid}.jpg) or, if the warehouse pull carried an
image_url (FA image-detection blob), downloads it once, downscales it to a small
JPEG thumbnail, and caches it under data/imgs so it is reused for both typing and
UI display. Types the outlet format; unusable / low-confidence photos are left
untyped so grading falls back to text. Local-only: needs torch + open_clip (kept
out of the slim serve image on purpose).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
# same persistent location the API serves photos from (PROFILER_DATA_DIR)
IMG_DIR = Path(os.environ.get("PROFILER_DATA_DIR", str(ROOT / "data"))) / "imgs"
_MODEL = None
_THUMB_MAX = 384        # cached thumbnail long edge — plenty for SigLIP (224) + display
_DL_WORKERS = 16
# outlet storefront photo host: the f2klocations imageid resolves here. Public CDN,
# overridable via PROFILER_IMAGE_URL_TEMPLATE for other deployments.
_DEFAULT_IMG_TMPL = "https://static.fieldassist.io/outletimages/{imageid}"


def available() -> bool:
    import importlib.util as u
    return bool(u.find_spec("torch")) and bool(u.find_spec("open_clip"))


def loaded() -> bool:
    """True once the SigLIP model has actually been loaded into memory (lazy —
    happens on the first photo job or an explicit warm)."""
    return _MODEL is not None


def _model():
    global _MODEL
    if _MODEL is None:
        import clip_classify
        _MODEL = (clip_classify, *clip_classify._load_model())
    return _MODEL


def _download_thumb(url: str, dst: Path) -> bool:
    """Download an image URL and write a downscaled JPEG thumbnail to dst."""
    import httpx
    from PIL import Image
    try:
        r = httpx.get(url, timeout=25, follow_redirects=True)
        r.raise_for_status()
        import io
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((_THUMB_MAX, _THUMB_MAX))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=82)
        return True
    except Exception:
        return False


def _resolve(outletid: int, ref: str | None) -> str | None:
    """Return a local path whose basename is {outletid}.jpg (clip_classify parses
    the id from the filename), or None if no image can be obtained. `ref` is a
    direct image URL (preferred) or a legacy imageid used with a CDN template."""
    p = IMG_DIR / f"{outletid}.jpg"
    if p.exists():
        return str(p)
    if ref and str(ref).startswith("http"):
        return str(p) if _download_thumb(str(ref), p) else None
    tmpl = os.environ.get("PROFILER_IMAGE_URL_TEMPLATE", _DEFAULT_IMG_TMPL)
    if ref and tmpl:
        try:
            url = tmpl.format(imageid=ref, outletid=outletid,
                              company=os.environ.get("PROFILER_IMAGE_COMPANY", ""))
        except Exception:
            return None
        return str(p) if _download_thumb(url, p) else None
    return None


def type_outlets(items: list[tuple[int, str | None]],
                 progress_cb: Callable[[int, int], None] | None = None,
                 fetch_cb: Callable[[int, int], None] | None = None,
                 batch: int = 32) -> tuple[dict[int, str], int]:
    """items: (outletid, image_url|imageid|None). Returns ({outletid: image_format},
    n_typed). Downloads+caches thumbnails concurrently (fetch_cb reports that),
    then classifies in batches (progress_cb reports that)."""
    cc, model, preprocess, labels, txt = _model()
    total = len(items)
    resolved: list[tuple[int, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=_DL_WORKERS) as ex:
        futs = {ex.submit(_resolve, oid, ref): oid for oid, ref in items}
        for fut in futs:
            oid = futs[fut]
            path = fut.result()
            if path:
                resolved.append((oid, path))
            done += 1
            if fetch_cb:
                fetch_cb(done, total)
    n_res = len(resolved)
    out: dict[int, str] = {}
    typed_done = 0
    for i in range(0, n_res, batch):
        chunk = resolved[i:i + batch]
        try:
            res = cc.classify(model, preprocess, labels, txt, [p for _, p in chunk])
            for oid, fmt, _conf, usable in res:
                if usable and fmt:
                    out[oid] = fmt
        except Exception:
            pass  # one bad batch can't kill onboarding
        typed_done += len(chunk)
        if progress_cb:
            progress_cb(typed_done, n_res)
    return out, n_res
