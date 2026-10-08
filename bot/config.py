import importlib.util
import subprocess
import sys
from pathlib import Path

if importlib.util.find_spec("pydantic_settings") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic-settings"])

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str
    log_level: str = "INFO"
    admin_ids: str = ""
    proxy_url: str = "socks5://127.0.0.1:10808"

    @property
    def admin_ids_set(self) -> set[int]:
        if not self.admin_ids:
            return set()
        return {int(x.strip()) for x in self.admin_ids.split(",") if x.strip().isdigit()}


settings = BotSettings()