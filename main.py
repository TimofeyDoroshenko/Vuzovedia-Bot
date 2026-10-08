import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

BOT_START_TIME = time.time()

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

VPS_MARKER = Path("/etc/sing-box/config.json")
IS_VPS = VPS_MARKER.exists()

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_formatter = logging.Formatter(_LOG_FMT)


def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_DIR / "bot.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(_formatter)
    root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "sentence_transformers",
                  "urllib3", "telegram", "asyncio", "aiohttp",
                  "aiogram.event"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger("vuz-bot")

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError
from aiogram.types import ErrorEvent

from bot.config import settings
from bot.rag import load_data_to_db
from bot.user_handlers import user_router


def _verify_rag_ready() -> None:
    from bot import rag
    emb = getattr(rag, "_embeddings", None)
    metas = getattr(rag, "_metadatas", None)
    if emb is None or metas is None or len(metas) == 0:
        raise RuntimeError("RAG не инициализирован: база пуста или не загрузилась.")
    logger.info("✅ RAG готов: %d документов, эмбеддинги %s",
                len(metas), tuple(emb.shape))


async def _on_error(event: ErrorEvent) -> bool:
    exc = event.exception
    if isinstance(exc, TelegramNetworkError):
        logger.warning("🌐 Сеть до Telegram моргнула (%s) — aiogram повторит.",
                       type(exc).__name__)
        return True
    logger.error("Необработанная ошибка в обработчике", exc_info=exc)
    return True


async def main() -> None:
    logger.info("🧠 Готовлю векторную базу...")
    try:
        await asyncio.to_thread(load_data_to_db)
    except Exception:
        logger.exception("❌ Не удалось инициализировать RAG — завершаю работу.")
        return

    _verify_rag_ready()

    if IS_VPS:
        session = AiohttpSession(proxy="socks5://127.0.0.1:10808")
        logger.info("🌐 Сессия через SOCKS5 (VPS-режим)")
    else:
        session = AiohttpSession()
        logger.info("🌐 Сессия напрямую (локальный режим)")

    bot = Bot(
        token=settings.bot_token,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(user_router)
    dp.errors.register(_on_error)

    # Антиспам-middleware: автобан за флуд, глобальный потолок, sanity-check длины.
    from bot.antispam import AntispamMiddleware
    antispam = AntispamMiddleware()
    dp.message.middleware(antispam)
    dp.callback_query.middleware(antispam)

    logger.info("🚀 Запускаю polling. Ctrl+C / SIGTERM — остановка.")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        logger.info("👋 Бот остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("👋 Пока!")