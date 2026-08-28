"""
Modul pro odstranění pozadí, barvení, inteligentní ořezávání (autocrop), měření času a rotaci monitoring logů.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from PIL import Image
from rembg import remove, new_session

# Konfigurace základního logování
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("img_cropper")

# Globální mezipaměť (cache) pro ONNX sessions, aby se model nenačítal znovu při každém volání
_SESSION_CACHE: dict = {}


@dataclass
class ProcessingMetrics:
    """Metriky měření času pro jeden obrázek."""
    input_file: str
    output_file: str
    total_time: float
    inference_time: float
    crop_time: float
    save_time: float
    image_size: Tuple[int, int]
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class BatchMetrics:
    """Souhrnné statistiky pro dávku (1..N) obrázků."""
    total_images: int
    successful: int
    failed: int
    total_time: float
    avg_time_per_image: float
    min_time: float
    max_time: float
    throughput_per_sec: float
    items: List[ProcessingMetrics]

    def summary(self) -> str:
        """Vrátí přehledný textový souhrn."""
        lines = [
            "=" * 65,
            "📊 SOUHRN MĚŘENÍ ČASU ZPRACOVÁNÍ",
            "=" * 65,
            f"Celkem obrázků:          {self.total_images} (Úspěšně: {self.successful}, Selhalo: {self.failed})",
            f"Celkový čas dávky:       {self.total_time:.2f} s",
            f"Průměrný čas na obrázek: {self.avg_time_per_image:.2f} s",
            f"Nejrychlejší obrázek:    {self.min_time:.2f} s",
            f"Nejpomalejší obrázek:    {self.max_time:.2f} s",
            f"Průchodnost:             {self.throughput_per_sec:.2f} obr/s (cca {self.throughput_per_sec * 60:.1f} obr/min)",
            "=" * 65,
        ]
        return "\n".join(lines)


def rotate_monitoring_logs(output_dir: Union[str, Path], max_files: int = 2) -> List[Path]:
    """
    Rotuje logy monitoringu v cílové složce.
    Zachovává vždy maximálně `max_files` nejnovějších souborů a starší maže.

    :param output_dir: Složka s logy
    :param max_files: Maximální počet uchovávaných log souborů (výchozí 2)
    :return: Seznam zbývajících zachovaných log souborů
    """
    out_dir_path = Path(output_dir)
    if not out_dir_path.exists():
        return []

    # Vyhledání všech monitoring souborů
    log_files = sorted(out_dir_path.glob("monitoring_*.log"), key=lambda f: f.name)

    # Pokud je souborů více než povolené maximum, smažeme nejstarší
    if len(log_files) > max_files:
        files_to_delete = log_files[:-max_files]
        for old_file in files_to_delete:
            try:
                old_file.unlink()
                logger.info(f"🗑️ Rotace logů: Odstraněn starý log '{old_file.name}'")
            except OSError as e:
                logger.warning(f"Nepodařilo se smazat starý log {old_file.name}: {e}")

    # Aktuální seznam po rotaci
    return sorted(out_dir_path.glob("monitoring_*.log"), key=lambda f: f.name)


def save_monitoring_log(
    metrics_data: Union[BatchMetrics, ProcessingMetrics, List[ProcessingMetrics]],
    output_dir: Union[str, Path] = "output",
    max_history: int = 2
) -> Path:
    """
    Zapíše výsledky monitoringu do nového log souboru číslovaného časovým razítkem
    a zajistí rotaci (ponechá pouze poslední 2 nejnovější soubory).

    :param metrics_data: Výsledky zpracování (BatchMetrics, ProcessingMetrics nebo list)
    :param output_dir: Cílová složka
    :param max_history: Počet uchovávaných logů (výchozí 2)
    :return: Cesta k vytvořenému log souboru
    """
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = out_dir_path / f"monitoring_{timestamp_str}.log"

    # Příprava položek k vypsání
    if isinstance(metrics_data, BatchMetrics):
        items = metrics_data.items
        batch_summary = metrics_data
    elif isinstance(metrics_data, ProcessingMetrics):
        items = [metrics_data]
        batch_summary = BatchMetrics(
            total_images=1,
            successful=1 if metrics_data.success else 0,
            failed=0 if metrics_data.success else 1,
            total_time=metrics_data.total_time,
            avg_time_per_image=metrics_data.total_time,
            min_time=metrics_data.total_time,
            max_time=metrics_data.total_time,
            throughput_per_sec=1 / metrics_data.total_time if metrics_data.total_time > 0 else 0,
            items=items
        )
    else:
        items = metrics_data
        valid_times = [m.total_time for m in items if m.success]
        t_total = sum(valid_times)
        batch_summary = BatchMetrics(
            total_images=len(items),
            successful=len(valid_times),
            failed=len(items) - len(valid_times),
            total_time=t_total,
            avg_time_per_image=t_total / len(valid_times) if valid_times else 0,
            min_time=min(valid_times) if valid_times else 0,
            max_time=max(valid_times) if valid_times else 0,
            throughput_per_sec=len(valid_times) / t_total if t_total > 0 else 0,
            items=items
        )

    # Tvorba obsahu logu
    lines = [
        "=" * 70,
        f"📋 MONITORING LOG ZPRACOVÁNÍ OBRÁZKŮ",
        f"Čas spuštění: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "--- DETAILNÍ PŘEHLED JEDNOTLIVÝCH POLOŽEK ---",
    ]

    for idx, item in enumerate(items, 1):
        if item.success:
            lines.append(
                f"[{idx:03d}] {Path(item.input_file).name} -> {Path(item.output_file).name} | "
                f"Rozlišení: {item.image_size[0]}x{item.image_size[1]} | "
                f"Celkem: {item.total_time:.3f}s (AI: {item.inference_time:.3f}s, "
                f"Ořez: {item.crop_time:.4f}s, Uložení: {item.save_time:.4f}s)"
            )
        else:
            lines.append(
                f"[{idx:03d}] {Path(item.input_file).name} -> CHYBA: {item.error_message}"
            )

    lines.extend([
        "",
        batch_summary.summary(),
        ""
    ])

    log_file_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"📝 Uložen monitoring log: '{log_file_path.name}'")

    # Rotace - ponechání pouze posledních N nejnovějších logů
    rotate_monitoring_logs(output_dir=out_dir_path, max_files=max_history)

    return log_file_path


def get_session(model_name: str = "birefnet-general"):
    """Získá existující nebo vytvoří novou session pro daný model."""
    if model_name not in _SESSION_CACHE:
        t0 = time.perf_counter()
        logger.info(f"Načítám ONNX model '{model_name}' do paměti...")
        _SESSION_CACHE[model_name] = new_session(model_name)
        logger.info(f"Model '{model_name}' načten za {time.perf_counter() - t0:.2f} s.")
    return _SESSION_CACHE[model_name]


def autocrop_transparent(
    image: Image.Image,
    padding: int = 0,
    aspect_ratio: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Automaticky ořízne průhledné okraje kolem objektu (podle alpha kanálu).
    
    :param image: Vstupní RGBA obrázek s průhledným pozadím
    :param padding: Odsazení v pixelech kolem oříznutého objektu
    :param aspect_ratio: Volitelný poměr stran (např. (1, 1) pro čtverec, (4, 5) atd.)
    :return: Oříznutý obrázek
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Získání ohraničujícího rámečku (bbox) z alpha kanálu
    alpha = image.split()[3]
    bbox = alpha.getbbox()

    if not bbox:
        return image

    left, top, right, bottom = bbox

    # Přidání paddingu
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)

    cropped = image.crop((left, top, right, bottom))

    # Pokud je vyžadován konkrétní poměr stran, provedeme vycentrování do cílového plátna
    if aspect_ratio:
        target_w_ratio, target_h_ratio = aspect_ratio
        current_w, current_h = cropped.size
        
        target_ratio = target_w_ratio / target_h_ratio
        current_ratio = current_w / current_h

        if current_ratio > target_ratio:
            new_w = current_w
            new_h = int(current_w / target_ratio)
        else:
            new_h = current_h
            new_w = int(current_h * target_ratio)

        canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
        offset_x = (new_w - current_w) // 2
        offset_y = (new_h - current_h) // 2
        canvas.paste(cropped, (offset_x, offset_y), mask=cropped)
        return canvas

    return cropped


def process_background(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    bg_color: Optional[Tuple[int, int, int, int]] = None,
    model_name: str = "birefnet-general",
    autocrop: bool = False,
    padding: int = 0,
    aspect_ratio: Optional[Tuple[int, int]] = None,
    quality: int = 90,
    log_time: bool = True
) -> Tuple[Image.Image, ProcessingMetrics]:
    """
    Hlavní funkce pro zpracování pozadí a volitelný ořez včetně měření času.

    :param input_path: Cesta ke zdrojovému obrázku
    :param output_path: Cesta pro uložení výsledku
    :param bg_color: RGBA/RGB tuple např. (255, 255, 255, 255) pro bílou, nebo None pro transparentní
    :param model_name: "birefnet-general" (velmi přesný), "u2net" (rychlý), "isnet-general-use" atd.
    :param autocrop: Pokud je True, automaticky ořízne prázdné okraje kolem hlavního objektu
    :param padding: Odsazení v pixelech při autocropu
    :param aspect_ratio: Cílový poměr stran při autocropu, např. (1, 1) pro čtverec
    :param quality: Kvalita komprese pro JPG/WebP (1-100, výchozí 90)
    :param log_time: Zda vypsat čas zpracování do logu
    :return: Tuple (výsledný PIL Image objekt, metriky ProcessingMetrics)
    """
    t_start = time.perf_counter()
    session = get_session(model_name)

    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)

    input_image = Image.open(input_path_obj)
    orig_size = input_image.size

    # 1. AI Segmentace (Inference)
    t_model_start = time.perf_counter()
    if autocrop or bg_color:
        output_image = remove(input_image, session=session)
    else:
        output_image = remove(input_image, session=session, bgcolor=bg_color)
    t_inference = time.perf_counter() - t_model_start

    # 2. Autocrop a geometrie
    t_crop_start = time.perf_counter()
    if autocrop:
        output_image = autocrop_transparent(
            output_image,
            padding=padding,
            aspect_ratio=aspect_ratio
        )

    if bg_color:
        if output_image.mode != "RGBA":
            output_image = output_image.convert("RGBA")
        background = Image.new("RGBA", output_image.size, bg_color)
        background.paste(output_image, (0, 0), mask=output_image)
        output_image = background
    t_crop = time.perf_counter() - t_crop_start

    # 3. Formátování a uložení
    t_save_start = time.perf_counter()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path_obj.suffix.lower()
    save_kwargs: dict = {}

    if ext in [".jpg", ".jpeg"]:
        save_kwargs["quality"] = quality
        if output_image.mode != "RGB":
            if bg_color is None:
                bg = Image.new("RGB", output_image.size, (255, 255, 255))
                bg.paste(output_image, mask=output_image.split()[3])
                output_image = bg
            else:
                output_image = output_image.convert("RGB")
    elif ext == ".webp":
        save_kwargs["quality"] = quality
        if bg_color and len(bg_color) == 4 and bg_color[3] == 255:
            output_image = output_image.convert("RGB")
    elif bg_color and len(bg_color) == 4 and bg_color[3] == 255 and ext not in [".png"]:
        output_image = output_image.convert("RGB")

    output_image.save(output_path_obj, **save_kwargs)
    t_save = time.perf_counter() - t_save_start

    t_total = time.perf_counter() - t_start

    metrics = ProcessingMetrics(
        input_file=str(input_path_obj),
        output_file=str(output_path_obj),
        total_time=t_total,
        inference_time=t_inference,
        crop_time=t_crop,
        save_time=t_save,
        image_size=orig_size
    )

    if log_time:
        logger.info(
            f"✅ [{input_path_obj.name} -> {output_path_obj.name}] "
            f"Hotovo za {t_total:.2f}s "
            f"(AI segmentace: {t_inference:.2f}s, ořez: {t_crop:.3f}s, zápis: {t_save:.3f}s)"
        )

    return output_image, metrics


def process_batch(
    images: List[Union[str, Path]],
    output_dir: Union[str, Path],
    output_format: str = "jpg",
    bg_color: Optional[Tuple[int, int, int, int]] = (255, 255, 255, 255),
    model_name: str = "birefnet-general",
    autocrop: bool = True,
    padding: int = 20,
    aspect_ratio: Optional[Tuple[int, int]] = (1, 1),
    save_log: bool = True,
    max_log_history: int = 2
) -> BatchMetrics:
    """
    Dávkové zpracování 1..N obrázků s detailním logováním času, souhrnnou statistikou
    a automatickým ukládáním i rotací monitoring logů.

    :param images: Seznam cest ke zdrojovým obrázkům
    :param output_dir: Složka pro uložení výsledků a logů
    :param output_format: Přípona výstupu ('jpg', 'png', 'webp')
    :param bg_color: Barva pozadí (nebo None pro průhlednost)
    :param model_name: Použitý model
    :param autocrop: Zda provádět ořez
    :param padding: Odsazení v px
    :param aspect_ratio: Poměr stran (např. (1, 1))
    :param save_log: Pokud je True, uloží monitoring log do output_dir
    :param max_log_history: Maximální počet uchovávaných log souborů (výchozí 2)
    :return: BatchMetrics se souhrnem a dílčími časy
    """
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    total_count = len(images)
    logger.info(f"🚀 Spouštím dávkové zpracování {total_count} obrázků (Model: {model_name})...")

    # Předehřátí / načtení modelu do paměti před startem měření dávky
    get_session(model_name)

    t_batch_start = time.perf_counter()
    metrics_list: List[ProcessingMetrics] = []
    successful = 0
    failed = 0

    for idx, img_path in enumerate(images, 1):
        p = Path(img_path)
        out_file = out_dir_path / f"{p.stem}_processed.{output_format.lstrip('.')}"
        logger.info(f"⏳ [{idx}/{total_count}] Zpracovávám: {p.name} ...")

        try:
            _, m = process_background(
                input_path=p,
                output_path=out_file,
                bg_color=bg_color,
                model_name=model_name,
                autocrop=autocrop,
                padding=padding,
                aspect_ratio=aspect_ratio,
                log_time=True
            )
            metrics_list.append(m)
            successful += 1
        except Exception as e:
            logger.error(f"❌ Chyba při zpracování {p.name}: {e}")
            failed += 1
            metrics_list.append(ProcessingMetrics(
                input_file=str(p),
                output_file=str(out_file),
                total_time=0.0,
                inference_time=0.0,
                crop_time=0.0,
                save_time=0.0,
                image_size=(0, 0),
                success=False,
                error_message=str(e)
            ))

    t_batch_total = time.perf_counter() - t_batch_start
    valid_times = [m.total_time for m in metrics_list if m.success]

    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0.0
    min_time = min(valid_times) if valid_times else 0.0
    max_time = max(valid_times) if valid_times else 0.0
    throughput = (successful / t_batch_total) if t_batch_total > 0 else 0.0

    batch_summary = BatchMetrics(
        total_images=total_count,
        successful=successful,
        failed=failed,
        total_time=t_batch_total,
        avg_time_per_image=avg_time,
        min_time=min_time,
        max_time=max_time,
        throughput_per_sec=throughput,
        items=metrics_list
    )

    print("\n" + batch_summary.summary() + "\n")

    if save_log:
        save_monitoring_log(batch_summary, output_dir=out_dir_path, max_history=max_log_history)

    return batch_summary
