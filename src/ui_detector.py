
from pathlib import Path

import cv2
import numpy as np

from src.log import Log


class UIDetector:

    def __init__(self):
        self.ui_template_dir = Path("assets/ui_templates")
        self.ui_templates: list[tuple[str, np.ndarray]] = []
        self._load_ui_templates()

    def _load_ui_templates(self) -> None:
        self.ui_templates.clear()

        if not self.ui_template_dir.exists():
            Log.warn(f"找不到 UI 模板文件夹: {self.ui_template_dir}")
            raise FileNotFoundError(
            )

        image_paths = []

        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths.extend(self.ui_template_dir.glob(suffix))

        if not image_paths:
            Log.error(f"模板文件夹中没有图片: {self.ui_template_dir}")
            raise FileNotFoundError(
            )

        for image_path in image_paths:
            template = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                Log.warn(f"无法读取 UI 模板: {image_path}")
                continue
            self.ui_templates.append((image_path.stem, template))
            Log.info(f"加载 UI 模板: {image_path}")

        if not self.ui_templates:
            Log.error("没有成功加载任何鱼图标模板。")
            raise RuntimeError(
            )
        
        Log.success(f"已加载 {len(self.ui_templates)} 个 UI 模板")

    