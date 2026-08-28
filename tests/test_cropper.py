import unittest
from pathlib import Path
import tempfile
import time
from PIL import Image, ImageDraw
from cropper import (
    autocrop_transparent,
    process_background,
    process_batch,
    save_monitoring_log,
    rotate_monitoring_logs,
    ProcessingMetrics,
    BatchMetrics
)


class TestImageCropper(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_autocrop_transparent(self):
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 30, 70, 70], fill=(255, 0, 0, 255))

        cropped = autocrop_transparent(img, padding=0)
        self.assertEqual(cropped.size, (41, 41))

        cropped_padded = autocrop_transparent(img, padding=5)
        self.assertEqual(cropped_padded.size, (51, 51))

        cropped_square = autocrop_transparent(img, padding=0, aspect_ratio=(1, 1))
        self.assertEqual(cropped_square.size[0], cropped_square.size[1])

    def test_process_background_and_timing(self):
        input_file = self.dir_path / "sample.png"
        output_file = self.dir_path / "output.png"
        
        img = Image.new("RGB", (60, 60), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([15, 15, 45, 45], fill=(0, 0, 0))
        img.save(input_file)

        res, metrics = process_background(
            input_path=input_file,
            output_path=output_file,
            bg_color=(255, 0, 0, 255),
            model_name="u2net",
            autocrop=True
        )

        self.assertTrue(output_file.exists())
        self.assertIsNotNone(res)
        self.assertIsInstance(metrics, ProcessingMetrics)
        self.assertGreater(metrics.total_time, 0.0)
        self.assertGreater(metrics.inference_time, 0.0)

    def test_log_rotation(self):
        # Vytvoříme uměle 4 log soubory s různými časovými razítky
        log1 = self.dir_path / "monitoring_20260827_100001.log"
        log2 = self.dir_path / "monitoring_20260827_100002.log"
        log3 = self.dir_path / "monitoring_20260827_100003.log"
        log4 = self.dir_path / "monitoring_20260827_100004.log"

        for p in [log1, log2, log3, log4]:
            p.write_text("dummy log content", encoding="utf-8")

        remaining = rotate_monitoring_logs(output_dir=self.dir_path, max_files=2)
        
        # Očekáváme přesně 2 zbývající nejnovější soubory
        self.assertEqual(len(remaining), 2)
        self.assertFalse(log1.exists())
        self.assertFalse(log2.exists())
        self.assertTrue(log3.exists())
        self.assertTrue(log4.exists())

    def test_batch_processing_and_monitoring_save(self):
        images = []
        for i in range(2):
            p = self.dir_path / f"batch_img_{i}.png"
            img = Image.new("RGB", (50, 50), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([10, 10, 30, 30], fill=(0, 0, 0))
            img.save(p)
            images.append(p)

        batch_metrics = process_batch(
            images=images,
            output_dir=self.dir_path,
            model_name="u2net",
            autocrop=True,
            save_log=True,
            max_log_history=2
        )

        self.assertIsInstance(batch_metrics, BatchMetrics)
        self.assertEqual(batch_metrics.total_images, 2)
        self.assertEqual(batch_metrics.successful, 2)

        # Ověření, že se vytvořil log soubor v cílové složce
        logs = list(self.dir_path.glob("monitoring_*.log"))
        self.assertEqual(len(logs), 1)
        content = logs[0].read_text(encoding="utf-8")
        self.assertIn("MONITORING LOG", content)


if __name__ == "__main__":
    unittest.main()
