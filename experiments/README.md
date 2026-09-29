# ACI watermark: LaMa experiment

`aci-watermark-mask.png` is a 900×450 mask of the repeated `www.aci.cz` watermark. It was estimated from the median red channel excess of 20 sample photos from `gs://circuparts-aci-img/aci/`, then dilated by 5 pixels. White pixels are replaced by LaMa; black pixels remain identical to the input. The mask is resized to each source image. It is specific to this ACI image layout.

Run with the repository's existing `.venv` (the `rembg[cpu]` dependency includes `onnxruntime`):

```bash
.venv/bin/python scripts/try_aci_lama.py --input output/wrecker-66/input --output-dir output/wrecker-66/lama
```

The script downloads the pinned 512×512 ONNX LaMa model once to `~/.cache/img-cropping/` and verifies its SHA-256. It saves each mask, PNG result and `metrics.json` under the output directory. Use `--limit 1` for a quick test or `--mask-template PATH` for another watermark layout. Inspect product details in every result; inpainting may invent or remove features under the mask.

For letters left around the edges, `--mask-grow 7` expands the mask by seven pixels in the 900×450 template before resizing. Try it on one image and save to a separate output directory:

```bash
.venv/bin/python scripts/try_aci_lama.py --input output/wrecker-66/input/113224_0.jpg --mask-grow 7 --output-dir output/wrecker-66/lama-grow7
```

The script records a layout warning when an image's aspect ratio differs from the template by more than 5%. The mask can be inspected under `masks/`.

Model: [sapienkit/LaMa-ONNX](https://huggingface.co/sapienkit/LaMa-ONNX), export of [LaMa](https://github.com/advimman/lama).
## Result on the 20 ACI samples (2026-09-29)

All 20 files completed on CPU in 52.02 s of per-image processing time. Inputs and results have identical dimensions, and pixels outside the saved mask are byte-for-byte unchanged. The comparison is in `output/wrecker-66/contact-original.jpg` and `output/wrecker-66/contact-lama.jpg` (local, ignored by Git).

The central watermark is much less visible on most standard 900×450 photos. This is still an experiment, not an automatic catalog replacement: the mask leaves red letter edges on `113224_0` (500×250), does not align across the whole watermark on `121489_0` (1500×682), and LaMa visibly reconstructs product texture on `101131_0` and `111280_0`. These files need manual review or a mask tailored to their layout.

Mask expansion by seven pixels removed the leftover letter edges on `113224_0`, but a full 19-image comparison showed more invented or blurred detail on the bumper, sheet metal, diagrams, packaging and lamps. It is therefore an opt-in adjustment, not the batch default. Enlarging and horizontally stretching the mask on `121489_0` damaged the barcode and mirror internals; this layout needs a separate approach.
