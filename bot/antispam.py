"""Анти-DDoS и анти-спам для бота.

Что защищаем:
  * флуд от одного юзера (автобан на 5 минут за 30+ событий в минуту);
  * флуд от многих юзеров одновременно (глобальный потолок 120/мин на весь бот);
  * засирание логов fallback-сообщениями (не чаще раза в минуту от юзера);
  * огромные сообщения (sanity-check длины).

Проверки выполняются в middleware, зарегистрированном в main.py, поэтому
новые хендлеры автоматически получают защиту и не могут её обойти.

Все пороги вынесены в константы и легко тюнятся.
"""

import time
from collections import deque

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from bot.config import settings

# ────── Пороги (легко тюнить) ──────

USER_MAX_EVENTS = 30            # событий от одного юзера
USER_WINDOW = 60.0              # за какой интервал (сек)
USER_BAN_DURATION = 300.0       # длительность бана (сек)

GLOBAL_LIMIT = 120              # суммарно событий от всех не-админов
GLOBAL_WINDOW = 60.0            # за какой интервал (сек)

MAX_INPUT_LEN = 200             # максимум символов в одном сообщении
FALLBACK_LOG_INTERVAL = 60.0    # как часто логировать/отвечать одному юзеру

# ────── Состояние ──────

_USER_BANS: dict[int, float] = {}           # user_id -> до какого timestamp забанен
_USER_ACTIVITY: dict[int, list[float]] = {} # user_id -> метки событий за окно
_GLOBAL_EVENTS: deque[float] = deque(maxlen=2000)
_THROTTLE_MARK: dict[int, float] = {}       # последнее сообщение юзеру о блокировке

# ────── Вспомогательные ──────

def _now() -> float:
    return time.time()


def is_banned(user_id: int, now: float | None = None) -> bool:
    if now is None:
        now = _now()
    ban_until = _USER_BANS.get(user_id)
    if ban_until is None:
        return False
    if now >= ban_until:
        _USER_BANS.pop(user_id, None)
        return False
    return True


def _register_event(user_id: int, now: float) -> str:
    """Возвращает 'ok' | 'just_banned' | 'banned'."""
    if is_banned(user_id, now):
        return "banned"

    stamps = _USER_ACTIVITY.get(user_id, [])
    stamps = [t for t in stamps if now - t < USER_WINDOW]
    stamps.append(now)
    _USER_ACTIVITY[user_id] = stamps

    if len(stamps) >= USER_MAX_EVENTS:
        _USER_BANS[user_id] = now + USER_BAN_DURATION
        _USER_ACTIVITY.pop(user_id, None)
        return "just_banned"
    return "ok"


def _global_event_ok(now: float) -> bool:
    while _GLOBAL_EVENTS and now - _GLOBAL_EVENTS[0] > GLOBAL_WINDOW:
        _GLOBAL_EVENTS.popleft()
    if len(_GLOBAL_EVENTS) >= GLOBAL_LIMIT:
        return False
    _GLOBAL_EVENTS.append(now)
    return True


def should_notify(user_id: int, now: float | None = None) -> bool:
    """Throttle для сообщений вида «вы забанены»: не чаще раза в минуту."""
    if now is None:
        now = _now()
    last = _THROTTLE_MARK.get(user_id, 0.0)
    if now - last < FALLBACK_LOG_INTERVAL:
        return False
    _THROTTLE_MARK[user_id] = now
    return True


def input_too_long(text: str) -> bool:
    return len(text or "") > MAX_INPUT_LEN


def register_explicit_violation(user_id: int) -> None:
    """Ручная регистрация нарушения из хендлера (например, мат в запросе)."""
    now = _now()
    _register_event(user_id, now)


def prune(now: float | None = None) -> None:
    """Чистка словарей. Вызывается из user_handlers._prune_rate_limit_caches."""
    if now is None:
        now = _now()

    stale_bans = [uid for uid, until in _USER_BANS.items() if until < now]
    for uid in stale_bans:
        _USER_BANS.pop(uid, None)

    stale_activity = [
        uid for uid, stamps in _USER_ACTIVITY.items()
        if not stamps or now - max(stamps) > USER_WINDOW * 2
    ]
    for uid in stale_activity:
        _USER_ACTIVITY.pop(uid, None)

    stale_throttle = [
        uid for uid, t in _THROTTLE_MARK.items()
        if now - t > FALLBACK_LOG_INTERVAL * 10
    ]
    for uid in stale_throttle:
        _THROTTLE_MARK.pop(uid, None)


# ────── Middleware ──────

async def _safe_reply(event: TelegramObject, text: str) -> None:
    """Отправляет ответ, не падая если что-то не так."""
    try:
        if isinstance(event, Message):
            await event.answer(text)
        elif isinstance(event, CallbackQuery):
            await event.answer()
    except Exception:
        pass


class AntispamMiddleware(BaseMiddleware):
    """Единая точка антиспама. Регистрируется в main.py для message и callback_query."""

    async def __call__(self, handler, event, data):
        user = data.get("event_from_user")

        # Админов не трогаем: они тестируют и не могут быть атакой.
        if user and user.id in settings.admin_ids_set:
            return await handler(event, data)

        now = _now()

        # 1) Проверка на бан конкретного юзера.
        if user:
            verdict = _register_event(user.id, now)
            if verdict == "just_banned":
                if should_notify(user.id, now):
                    await _safe_reply(
                        event,
                        "⛔ Слишком много запросов. Вы заблокированы на 5 минут.",
                    )
                return
            if verdict == "banned":
                if should_notify(user.id, now):
                    await _safe_reply(
                        event,
                        "⛔ Вы временно заблокированы. Попробуйте через несколько минут.",
                    )
                return

        # 2) Глобальный потолок (защита от ботнета).
        if user and not _global_event_ok(now):
            if should_notify(user.id, now):
                await _safe_reply(
                    event,
                    "🌐 Бот перегружен. Попробуйте через минуту.",
                )
            return

        # 3) Sanity-check длины (только для сообщений).
        if isinstance(event, Message) and input_too_long(event.text or ""):
            await _safe_reply(
                event,
                "❌ Сообщение слишком длинное. Уложитесь в 200 символов.",
            )
            return

        return await handler(event, data)