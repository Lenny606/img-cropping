---
name: aci-watermark-lama
description: Use when testing or tuning ACI watermark removal with LaMa in img-cropping - vodoznak ACI, wrecker 66, maska, inpainting.
---

# ACI watermark: LaMa

Použij tento skill pro zkoušku odstranění vodoznaku `www.aci.cz` na fotkách ACI (`wrecker_id = 66`) v tomto repozitáři. Pro běžné odstranění pozadí bez vodoznaku použij `cropper.py` přímo.

## Postup

1. Pracuj z kořene repozitáře. Před zpracováním zjisti cestu k požadovaným originálům a zkontroluj pracovní strom. Vzorové fotky, pokud jsou lokálně k dispozici, leží v `output/wrecker-66/input/`; tento adresář je ignorovaný Gitem. Zdroj vzorku je `gs://circuparts-aci-img/aci/`.
2. Přečti `experiments/README.md`, pokud měníš masku nebo vyhodnocuješ kvalitu. Skript je `scripts/try_aci_lama.py`, výchozí maska `experiments/aci-watermark-mask.png`.
3. Spusť nejprve jeden reprezentativní snímek do vlastního adresáře, například:

   ```bash
   .venv/bin/python scripts/try_aci_lama.py --input output/wrecker-66/input/101131_0.jpg --output-dir output/wrecker-66/lama-check
   ```

   Pokud `.venv` chybí, vytvoř ho a nainstaluj `requirements.txt`. Skript stahuje ONNX model do uživatelské cache a ověřuje jeho SHA-256.
4. Při přijatelném výsledku spusť požadovanou dávku přes `--input` (soubor nebo adresář) a `--output-dir`. Každou variantu masky ukládej do jiného adresáře. `--mask-grow N` zkoušej jen na snímcích, kde zůstávají okraje písmen; plošně větší maska rozmazávala detaily. Pro jiné rozložení vodoznaku použij `--mask-template PATH`.

## Kontrola výsledku

- Zkontroluj `metrics.json`, počet úspěšných souborů a uložené masky v `masks/`. Varování o odlišném poměru stran znamená, že výchozí maska nemusí sedět.
- Vizuálně porovnej originál, masku a výsledek zejména na štítcích, čárových kódech, reflektorech, mřížkách a tenkých dílech. LaMa dopočítává pixely pod maskou a může změnit skutečnou geometrii nebo text.
- Uveď, které snímky jsou použitelné a které vyžadují jinou masku nebo ruční kontrolu. Výstupy experimentu jsou lokální; do katalogu ani GCS je nenahrávej bez požadavku uživatele.
