"""Try LaMa inpainting on ACI product photos with a fixed watermark mask.

The model fills only masked pixels. Source pixels outside the mask are copied
unchanged to the output. Inspect every result before using it in a catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = "https://huggingface.co/sapienkit/LaMa-ONNX/resolve/main/lama_fp32.onnx"
MODEL_SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
MODEL_SIZE = (512, 512)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == MODEL_SHA256:
        return path
    temporary = path.with_suffix(path.suffix + ".download")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded LaMa model has an unexpected SHA-256 hash")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_mask(template: Image.Image, size: tuple[int, int], grow: int = 0) -> Image.Image:
    if grow:
        template = template.filter(ImageFilter.MaxFilter(2 * grow + 1))
    mask = template.resize(size, Image.Resampling.BILINEAR)
    return mask.point(lambda value: 255 if value >= 128 else 0, mode="L")


def inpaint(session: ort.InferenceSession, image: Image.Image, mask: Image.Image) -> Image.Image:
    small_image = image.resize(MODEL_SIZE, Image.Resampling.LANCZOS)
    small_mask = mask.resize(MODEL_SIZE, Image.Resampling.NEAREST)
    image_array = np.asarray(small_image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    mask_array = (np.asarray(small_mask, dtype=np.float32)[None, None] >= 128).astype(np.float32)
    output = session.run(None, {"image": image_array, "mask": mask_array})[0]
    rgb = np.clip(output[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    generated = Image.fromarray(rgb, "RGB").resize(image.size, Image.Resampling.BICUBIC)
    return Image.composite(generated, image, mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/wrecker-66/input",
                        help="Image or directory of images to process")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/wrecker-66/lama")
    parser.add_argument("--mask-template", type=Path, default=ROOT / "experiments/aci-watermark-mask.png")
    parser.add_argument("--model", type=Path, default=Path.home() / ".cache/img-cropping/lama_fp32.onnx")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images")
    parser.add_argument("--mask-grow", type=int, default=0,
                        help="Expand the template by this many pixels before resizing (default: 0)")
    args = parser.parse_args()
    if args.mask_grow < 0:
        parser.error("--mask-grow must be nonnegative")

    images = ([args.input] if args.input.is_file() else
              sorted(path for path in args.input.iterdir() if path.suffix.lower() in EXTENSIONS))
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error(f"No images found at {args.input}")
    template = Image.open(args.mask_template).convert("L")
    model_path = ensure_model(args.model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "masks").mkdir(exist_ok=True)

    results = []
    for index, path in enumerate(images, 1):
        started = time.perf_counter()
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            mask = make_mask(template, image.size, args.mask_grow)
            result = inpaint(session, image, mask)
            output = args.output_dir / f"{path.stem}_lama.png"
            result.save(output)
            mask.save(args.output_dir / "masks" / f"{path.stem}_mask.png")
            aspect_difference = abs((image.width / image.height) / (template.width / template.height) - 1)
            item = {"input": str(path), "output": str(output), "success": True,
                    "seconds": round(time.perf_counter() - started, 3),
                    "layout_warning": aspect_difference > 0.05}
        except Exception as exc:
            item = {"input": str(path), "success": False, "error": str(exc)}
        results.append(item)
        print(f"[{index}/{len(images)}] {path.name}: " +
              ((f"{item['seconds']} s" + (" (different aspect ratio; inspect mask)" if item["layout_warning"] else ""))
               if item["success"] else f"ERROR {item['error']}"), flush=True)

    (args.output_dir / "metrics.json").write_text(
        json.dumps({"model_url": MODEL_URL, "model_sha256": MODEL_SHA256,
                    "mask_template": str(args.mask_template), "mask_grow": args.mask_grow,
                    "items": results}, indent=2),
        encoding="utf-8")
    if any(not item["success"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
