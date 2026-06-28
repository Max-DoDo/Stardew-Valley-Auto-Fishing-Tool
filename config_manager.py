import json
from pathlib import Path
from typing import Any

class ConfigManager:

    def __init__(self, config_path: str | Path = "config.json"):
        self.config_path = Path(config_path)

    def load(self) -> dict[str, Any]:
        """
        从 JSON 文件读取配置。
        如果文件不存在，返回默认配置。
        """
        config = {}

        if not self.config_path.exists():
            return config

        try:
            text = self.config_path.read_text(encoding="utf-8")
            user_config = json.loads(text) if text.strip() else {}

        except json.JSONDecodeError as e:
            raise ValueError(f"Config file JSON type error{self.config_path}") from e

        if not isinstance(user_config, dict):
            raise ValueError("Config file must contain JSON object")

        self._deep_update(config, user_config)

        return config

    def update(self, values: dict[str, Any]) -> None:
        """
        更新 JSON 配置文件。
        """
        config = self.load()

        self._deep_update(config, values)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.config_path.write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )

    def _deep_update(self, old: dict, new: dict) -> dict:
        """
        递归合并字典。
        """
        for key, value in new.items():
            if (
                key in old
                and isinstance(old[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(old[key], value)
            else:
                old[key] = value

        return old