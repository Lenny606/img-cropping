#!/usr/bin/env python3
"""
Hlavní spouštěcí skript s ukázkou měření času pro 1 až N obrázků.
"""

import sys
from pathlib import Path

try:
    from cropper import process_background, process_batch, save_monitoring_log
except ModuleNotFoundError as e:
    print("\n" + "!" * 65)
    print("⚠️  CHYBA: Balíčky nejsou v systémovém Pythonu nainstalovány.")
    print("👉 Spusťte skript pomocí virtuálního prostředí:")
    print("   1) source .venv/bin/activate && python3 main.py")
    print("   NEBO přímo:")
    print("   2) .venv/bin/python3 main.py")
    print("!" * 65 + "\n")
    sys.exit(1)


def run_examples():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    input_file = "img-src.jpg"

    if not Path(input_file).exists():
        print(f"Chyba: Vstupní soubor '{input_file}' nenalezen.")
        return

    print("=" * 60)
    print("1️⃣  MĚŘENÍ ČASU PRO 1 OBRÁZEK")
    print("=" * 60)

    # 1. Zpracování do JPG s bílým pozadím
    _, metrics_jpg = process_background(
        input_path=input_file,
        output_path=output_dir / "output_white.jpg",
        bg_color=(255, 255, 255, 255),
        model_name="birefnet-general",
        autocrop=True,
        padding=20,
        aspect_ratio=(1, 1),
        quality=90
    )

    # 2. Zpracování do moderního WebP s bílým pozadím
    _, metrics_webp = process_background(
        input_path=input_file,
        output_path=output_dir / "output_white.webp",
        bg_color=(255, 255, 255, 255),
        model_name="birefnet-general",
        autocrop=True,
        padding=20,
        aspect_ratio=(1, 1),
        quality=90
    )

    print(f"\nVýsledek pro WebP výstup:")
    print(f" - Celkový čas:      {metrics_webp.total_time:.2f} s")
    print(f" - Z toho AI model:  {metrics_webp.inference_time:.2f} s")
    print(f" - Z toho ořez/crop: {metrics_webp.crop_time:.4f} s")
    print(f" - Z toho ukládání:  {metrics_webp.save_time:.4f} s")

    # Uložení monitoring logu pro jednotlivé obrázky
    save_monitoring_log([metrics_jpg, metrics_webp], output_dir=output_dir, max_history=2)

    print("\n" + "=" * 60)
    print("2️⃣  MĚŘENÍ ČASU PRO DÁVKU N OBRÁZKŮ (Simulace 3 položek)")
    print("=" * 60)

    # Vytvoříme seznam cest k obrázkům pro ukázku dávky N položek
    batch_inputs = [input_file, input_file, input_file]

    batch_metrics = process_batch(
        images=batch_inputs,
        output_dir=output_dir / "batch",
        output_format="jpg",
        bg_color=(255, 255, 255, 255),
        model_name="birefnet-general",
        autocrop=True,
        padding=20,
        aspect_ratio=(1, 1)
    )


if __name__ == "__main__":
    run_examples()
