import asyncio
import html
import logging
import re
import sqlite3
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    LinkPreviewOptions,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from bot import antispam
from bot.config import BASE_DIR, settings
from bot.rag import format_one_vuz, get_index_stats, get_proper_city_name, load_data_to_db, search_vuz

logger = logging.getLogger(__name__)

user_router = Router()

UPTIME_START = time.time()

MSK = timezone(timedelta(hours=3))

# ───── БД Аналитики и Избранного ─────
DB_PATH = BASE_DIR / "analytics.db"
with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS queries ("
        "id INTEGER PRIMARY KEY, "
        "user_id INTEGER, "
        "city TEXT, "
        "specialty TEXT, "
        "score INTEGER, "
        "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS favorites ("
        "id INTEGER PRIMARY KEY, "
        "user_id INTEGER NOT NULL, "
        "university TEXT NOT NULL, "
        "url TEXT, "
        "city TEXT, "
        "directions TEXT, "
        "budget TEXT, "
        "paid TEXT, "
        "saved_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
        "UNIQUE(user_id, university))"
    )

try:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("ALTER TABLE queries ADD COLUMN is_admin INTEGER DEFAULT 0")
except sqlite3.OperationalError:
    pass


class SearchCriteria(StatesGroup):
    city = State()
    specialty = State()
    score = State()


# ───── Клавиатуры ─────
cancel_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Отмена")]],
    resize_keyboard=True,
)
city_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Любой")], [KeyboardButton(text="Отмена")]],
    resize_keyboard=True,
)

HOME_ROW = [InlineKeyboardButton(text="🏠 Меню", callback_data="open_menu")]


def kb_with_home(rows: list) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=list(rows) + [HOME_ROW])


def build_main_menu(user_id: int) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="🎯 Начать подбор", callback_data="start_task")],
        [
            InlineKeyboardButton(text="⭐ Избранное", callback_data="open_favorites"),
            InlineKeyboardButton(text="🕐 История", callback_data="open_history"),
        ],
        [InlineKeyboardButton(text="ℹ️ Помощь", callback_data="open_help")],
    ]
    if user_id in settings.admin_ids_set:
        rows.append([InlineKeyboardButton(text="📊 Аналитика", callback_data="analytics:all")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ───── Состояние и лимиты ─────
_RESULTS_CACHE = OrderedDict()
_MAX_RESULTS_CACHE = 100

_USER_LAST_SEARCH: dict[int, float] = {}
SEARCH_COOLDOWN_SEC = 5.0

_TASK_TIMESTAMPS: dict[int, list[float]] = {}
TASK_BURST_LIMIT = 3
TASK_WINDOW_SEC = 10.0

_PRUNE_INTERVAL_SEC = 300.0
_LAST_PRUNE_TIME = 0.0
_USER_LAST_SEARCH_TTL = 3600.0

_LAST_MENU_MSG: dict[int, int] = {}

_SKIP_SPECIALTY = {
    "любой", "любая", "любое", "любые",
    "все", "всё",
    "неважно", "не важно",
    "без разницы",
}

# Технические термины, которые пользователи пишут латиницей.
# Всё остальное на латинице по-прежнему отклоняется.
_EN_TECH_TERMS = {
    "it", "qa", "ai", "ml", "web", "data", "sql",
    "python", "java", "javascript", "kotlin", "swift", "golang", "go",
    "frontend", "backend", "fullstack", "devops",
    "ux", "ui", "c++", "c#",
    "data science", "data-science", "datascience", "datascientist",
}

# Стоп-корни для нецензурной и оскорбительной лексики.
# Проверка идёт по подстроке, список короткий.
_STOP_WORDS = {
    "хуй", "хуе", "хуё", "хуи", "хую", "хуя",
    "пизд", "бляд", "блят",
    "ебат", "ебал", "ебуч", "ебыр", "ебан", "ебну", "ёб",
    "ебля", "ебли",
    "сучар", "сучк", "сукин",
    "манда", "манде", "манду",
    "дроч",
    "гандон", "гондон",
    "шлюх", "шлюш",
    "анус",
    "пенис", "пениса", "пенисы", "пенису",
    "вагин",
    "клитор",
    "оргазм",
    "сперм",
    "порн",
    "онан",
    "минет", "миньет",
    "куни", "кунни",
    "мраз", "мрази",
    "твар", "твари",
    "ублюд",
    "нищеброд",
    "быдл",
    "дебил", "дибил",
    "идиот",
    "кретин",
    "козел", "козёл", "козл",
}

MAINTENANCE_MODE = False


def is_maintenance(user_id: int) -> bool:
    return MAINTENANCE_MODE and user_id not in settings.admin_ids_set


async def _delete_last_menu(message: Message, user_id: int) -> None:
    prev_id = _LAST_MENU_MSG.pop(user_id, None)
    if prev_id is None:
        return
    try:
        await message.bot.delete_message(chat_id=message.chat.id, message_id=prev_id)
    except Exception:
        pass


async def send_menu(message: Message, user_id: int, text: str, kb: InlineKeyboardMarkup) -> None:
    await _delete_last_menu(message, user_id)
    sent = await message.answer(
        text,
        reply_markup=kb,
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )
    _LAST_MENU_MSG[user_id] = sent.message_id


def _prune_rate_limit_caches(now: float) -> None:
    global _LAST_PRUNE_TIME
    if now - _LAST_PRUNE_TIME < _PRUNE_INTERVAL_SEC:
        return
    _LAST_PRUNE_TIME = now

    stale_users = [uid for uid, t in _USER_LAST_SEARCH.items() if now - t > _USER_LAST_SEARCH_TTL]
    for uid in stale_users:
        _USER_LAST_SEARCH.pop(uid, None)

    stale_task_users = []
    for uid, ts in _TASK_TIMESTAMPS.items():
        fresh = [t for t in ts if now - t < TASK_WINDOW_SEC]
        if fresh:
            _TASK_TIMESTAMPS[uid] = fresh
        else:
            stale_task_users.append(uid)
    for uid in stale_task_users:
        _TASK_TIMESTAMPS.pop(uid, None)

    # Чистим словари антиспама (баны, активность, throttle).
    antispam.prune(now)


def check_task_rate_limit(user_id: int) -> bool:
    if user_id in settings.admin_ids_set:
        return True
    now = time.time()
    _prune_rate_limit_caches(now)

    ts = _TASK_TIMESTAMPS.get(user_id, [])
    ts = [t for t in ts if now - t < TASK_WINDOW_SEC]
    if len(ts) >= TASK_BURST_LIMIT:
        _TASK_TIMESTAMPS[user_id] = ts
        return False
    ts.append(now)
    _TASK_TIMESTAMPS[user_id] = ts
    return True


def _cache_put(query_info: tuple, metas: list) -> str:
    key = uuid.uuid4().hex[:12]
    _RESULTS_CACHE[key] = (query_info[0], query_info[1], query_info[2], metas)
    while len(_RESULTS_CACHE) > _MAX_RESULTS_CACHE:
        _RESULTS_CACHE.popitem(last=False)
    return key


def _cache_get(key: str):
    return _RESULTS_CACHE.get(key)


def format_score(score: int) -> str:
    if score % 100 in (11, 12, 13, 14):
        return f"{score} баллов"
    last = score % 10
    if last == 1:
        return f"{score} балл"
    if last in (2, 3, 4):
        return f"{score} балла"
    return f"{score} баллов"


# ───── Клавиатура страницы результатов ─────
def get_page_kb(key: str, page: int, total_pages: int,
                page_size: int, offset: int) -> InlineKeyboardMarkup:
    rows = []

    if page_size > 0:
        fav_row = []
        for i in range(page_size):
            idx = offset + i
            fav_row.append(InlineKeyboardButton(
                text=f"⭐ {offset + i + 1}",
                callback_data=f"fav:{key}:{idx}",
            ))
        rows.append(fav_row)

    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"page:{key}:{page-1}"))
        nav.append(InlineKeyboardButton(text=f"{page+1}/{total_pages}", callback_data="ignore"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(text="Вперед ➡️", callback_data=f"page:{key}:{page+1}"))
        rows.append(nav)

    return kb_with_home(rows)


async def send_page(message_or_callback, key: str, page: int, user_id: int):
    data = _cache_get(key)
    if not data:
        msg = "⌛ Список устарел, выполните поиск заново."
        if isinstance(message_or_callback, CallbackQuery):
            await message_or_callback.message.answer(msg, reply_markup=build_main_menu(user_id))
        else:
            await message_or_callback.answer(msg, reply_markup=build_main_menu(user_id))
        return

    city_disp, spec_disp, score, metas = data

    if not metas:
        msg = "К сожалению, по вашим критериям ничего не найдено."
        if isinstance(message_or_callback, CallbackQuery):
            await message_or_callback.message.answer(msg, reply_markup=build_main_menu(user_id))
        else:
            await message_or_callback.answer(msg, reply_markup=build_main_menu(user_id))
        return

    header = f"🔎 Запрос: {html.escape(city_disp)} | {html.escape(spec_disp)} | {format_score(score)}\n\n"

    ITEMS_PER_PAGE = 5
    total_pages = (len(metas) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    page = max(0, min(page, total_pages - 1))

    start_idx = page * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    page_metas = metas[start_idx:end_idx]

    text = header + f"🎓 Найдено вариантов: {len(metas)}\n\n"
    numbered = []
    for i, m in enumerate(page_metas):
        numbered.append(f"<b>{start_idx + i + 1}.</b> {format_one_vuz(m)}")
    text += "\n\n".join(numbered)

    kb = get_page_kb(key, page, total_pages, len(page_metas), start_idx)

    try:
        if isinstance(message_or_callback, CallbackQuery):
            await message_or_callback.message.edit_text(
                text,
                reply_markup=kb,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
        else:
            await message_or_callback.answer(
                text,
                reply_markup=kb,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
    except TelegramBadRequest as e:
        if "not modified" in str(e).lower():
            return
        logger.warning("TelegramBadRequest при отправке страницы: %s", e)


# ───── Аналитика ─────
def _build_analytics_text(period: str) -> str | None:
    if period == "week":
        where = "WHERE is_admin = 0 AND timestamp >= datetime('now', '-7 days')"
        title = "📊 <b>Аналитика за 7 дней (МСК)</b>\n"
    else:
        where = "WHERE is_admin = 0"
        title = "📊 <b>Аналитика запросов (МСК)</b>\n"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute(f"SELECT COUNT(*) AS n FROM queries {where}")
            total = cur.fetchone()["n"]
            if total == 0:
                return None

            cur.execute(f"SELECT COUNT(DISTINCT user_id) AS n FROM queries {where}")
            unique_users = cur.fetchone()["n"]

            cur.execute(
                "SELECT COUNT(*) AS n FROM queries "
                "WHERE is_admin = 0 "
                "AND date(timestamp, '+3 hours') = date('now', '+3 hours')"
            )
            today = cur.fetchone()["n"]

            cur.execute(f"SELECT AVG(score) AS a FROM queries {where}")
            avg_score = cur.fetchone()["a"] or 0

            cur.execute(
                f"SELECT LOWER(city) AS c, COUNT(*) AS n "
                f"FROM queries {where} "
                f"GROUP BY LOWER(city) ORDER BY n DESC LIMIT 5"
            )
            top_cities = cur.fetchall()

            cur.execute(
                f"SELECT LOWER(specialty) AS s, COUNT(*) AS n "
                f"FROM queries {where} "
                f"GROUP BY LOWER(specialty) ORDER BY n DESC LIMIT 5"
            )
            top_specs = cur.fetchall()

            cur.execute(
                f"SELECT user_id, city, specialty, score, "
                f"datetime(timestamp, '+3 hours') AS ts_msk "
                f"FROM queries {where} ORDER BY id DESC LIMIT 5"
            )
            recent = cur.fetchall()

    except Exception:
        logger.exception("Ошибка при чтении аналитики")
        return None

    def short(text, limit=40):
        text = str(text)
        return text if len(text) <= limit else text[:limit - 1] + "…"

    lines = [title]
    lines.append(f"Всего запросов: <code>{total}</code>")
    lines.append(f"Уникальных юзеров: <code>{unique_users}</code>")
    lines.append(f"Сегодня: <code>{today}</code>")
    lines.append(f"Средний балл: <code>{avg_score:.0f}</code>")

    lines.append("\n<b>🏙 Топ-5 городов:</b>")
    for row in top_cities:
        city = html.escape(short(row["c"]).capitalize())
        lines.append(f"  • {city} — {row['n']}")

    lines.append("\n<b>📚 Топ-5 направлений:</b>")
    for row in top_specs:
        spec = html.escape(short(row["s"]).capitalize())
        lines.append(f"  • {spec} — {row['n']}")

    lines.append("\n<b>🕐 Последние 5 запросов:</b>")
    for row in recent:
        city = html.escape(short(row["city"], 20))
        spec = html.escape(short(row["specialty"], 25))
        lines.append(
            f"  • <code>{html.escape(row['ts_msk'])}</code> | "
            f"uid={row['user_id']} | {city} / {spec} / {row['score']}"
        )

    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:4000] + "\n…сообщение обрезано"
    return text


def _analytics_kb(period: str) -> InlineKeyboardMarkup:
    if period == "week":
        switch = InlineKeyboardButton(text="⬅️ Всё время", callback_data="analytics:all")
    else:
        switch = InlineKeyboardButton(text="📅 За 7 дней", callback_data="analytics:week")
    refresh = InlineKeyboardButton(text="🔄 Обновить", callback_data=f"analytics:{period}")
    return kb_with_home([[switch, refresh]])


# ───── Избранное ─────
FAVORITES_PER_PAGE = 5


def _build_favorites_view(user_id: int, page: int = 0):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) AS n FROM favorites WHERE user_id = ?",
                (user_id,),
            )
            total = cur.fetchone()["n"]
            if total == 0:
                return None, None, 0

            total_pages = (total + FAVORITES_PER_PAGE - 1) // FAVORITES_PER_PAGE
            page = max(0, min(page, total_pages - 1))
            offset = page * FAVORITES_PER_PAGE

            cur.execute(
                "SELECT id, university, url, city FROM favorites "
                "WHERE user_id = ? ORDER BY saved_at DESC, id DESC "
                "LIMIT ? OFFSET ?",
                (user_id, FAVORITES_PER_PAGE, offset),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Ошибка чтения favorites")
        return None, None, 0

    lines = [f"⭐ <b>Избранное</b> — всего {total}\n"]
    for i, r in enumerate(rows):
        num = offset + i + 1
        univ = html.escape(r["university"])
        url = html.escape(r["url"] or "#", quote=True)
        city = html.escape(str(r["city"] or ""))
        lines.append(f'{num}. 🏛 <a href="{url}">{univ}</a> — {city}')

    buttons = []
    row = []
    for i, r in enumerate(rows):
        num = offset + i + 1
        row.append(InlineKeyboardButton(
            text=f"🗑 {num}",
            callback_data=f"favdel:{r['id']}:{page}",
        ))
    if row:
        buttons.append(row)

    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"favpage:{page-1}"))
        nav.append(InlineKeyboardButton(text=f"{page+1}/{total_pages}", callback_data="ignore"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(text="Вперед ➡️", callback_data=f"favpage:{page+1}"))
        buttons.append(nav)

    return "\n".join(lines), kb_with_home(buttons), total_pages


# ───── История ─────
HISTORY_PER_PAGE = 5


def _build_history_view(user_id: int, page: int = 0):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) AS n FROM queries WHERE user_id = ?",
                (user_id,),
            )
            total = cur.fetchone()["n"]
            if total == 0:
                return None, None, 0

            total_pages = (total + HISTORY_PER_PAGE - 1) // HISTORY_PER_PAGE
            page = max(0, min(page, total_pages - 1))
            offset = page * HISTORY_PER_PAGE

            cur.execute(
                "SELECT id, city, specialty, score, "
                "datetime(timestamp, '+3 hours') AS ts_msk "
                "FROM queries WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (user_id, HISTORY_PER_PAGE, offset),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Ошибка чтения истории")
        return None, None, 0

    lines = [f"🕐 <b>История поисков</b> — всего {total}\n"]
    for i, r in enumerate(rows):
        num = offset + i + 1
        city = html.escape(str(r["city"])[:30])
        spec = html.escape(str(r["specialty"])[:30])
        lines.append(
            f"<b>{num}.</b> <code>{html.escape(r['ts_msk'])}</code>\n"
            f"     {city} / {spec} / {r['score']}"
        )

    buttons = []
    row = []
    for i, r in enumerate(rows):
        num = offset + i + 1
        row.append(InlineKeyboardButton(
            text=f"🔁 {num}",
            callback_data=f"repeat:{r['id']}",
        ))
    if row:
        buttons.append(row)

    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"histpage:{page-1}"))
        nav.append(InlineKeyboardButton(text=f"{page+1}/{total_pages}", callback_data="ignore"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton(text="Вперед ➡️", callback_data=f"histpage:{page+1}"))
        buttons.append(nav)

    return "\n\n".join(lines), kb_with_home(buttons), total_pages


# ───── Админские команды ─────
@user_router.message(Command("maintenance"))
async def cmd_maintenance(message: Message):
    if message.from_user.id not in settings.admin_ids_set:
        return
    global MAINTENANCE_MODE
    MAINTENANCE_MODE = not MAINTENANCE_MODE
    status = "ВКЛЮЧЕН" if MAINTENANCE_MODE else "ВЫКЛЮЧЕН"
    await message.answer(f"🛠 Режим обслуживания {status}.")


@user_router.message(Command("log"))
async def cmd_log(message: Message):
    if message.from_user.id not in settings.admin_ids_set:
        return
    log_path = BASE_DIR / "logs" / "bot.log"
    if log_path.exists():
        await message.answer_document(FSInputFile(log_path))
    else:
        await message.answer("📄 Файл логов не найден.")


@user_router.message(Command("stats"))
async def cmd_stats(message: Message):
    if message.from_user.id not in settings.admin_ids_set:
        return

    uptime_sec = int(time.time() - UPTIME_START)
    uptime_str = f"{uptime_sec // 3600}ч {(uptime_sec % 3600) // 60}м"

    stats = get_index_stats()
    dt = datetime.fromtimestamp(stats["last_index_time"], tz=MSK).strftime("%Y-%m-%d %H:%M:%S")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(id) FROM queries")
            total_queries = cur.fetchone()[0]
            cur.execute("SELECT COUNT(id) FROM favorites")
            total_favs = cur.fetchone()[0]
    except Exception:
        total_queries = 0
        total_favs = 0

    await message.answer(
        f"📊 <b>Статистика бота:</b>\n"
        f"Аптайм: <code>{uptime_str}</code>\n"
        f"Документов в базе: <code>{stats['count']}</code>\n"
        f"Последняя индексация: <code>{dt}</code>\n"
        f"Записей в кэше выдачи: <code>{len(_RESULTS_CACHE)} / {_MAX_RESULTS_CACHE}</code>\n"
        f"Всего запросов в БД: <code>{total_queries}</code>\n"
        f"Избранных вузов: <code>{total_favs}</code>",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_with_home([]),
    )


@user_router.message(Command("analytics"))
async def cmd_analytics(message: Message):
    if message.from_user.id not in settings.admin_ids_set:
        return
    text = _build_analytics_text("all")
    if text is None:
        await send_menu(
            message,
            message.from_user.id,
            "📊 Аналитика пуста — запросов ещё не было.",
            build_main_menu(message.from_user.id),
        )
        return
    await message.answer(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=_analytics_kb("all"),
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )


@user_router.callback_query(F.data.startswith("analytics:"))
async def cb_analytics(callback: CallbackQuery):
    if callback.from_user.id not in settings.admin_ids_set:
        await callback.answer("⛔ Нет доступа.", show_alert=True)
        return

    period = callback.data.split(":", 1)[1]
    if period not in ("all", "week"):
        await callback.answer()
        return

    text = _build_analytics_text(period)
    await callback.answer()

    if text is None:
        try:
            await callback.message.edit_text(
                "📊 За выбранный период запросов нет.",
                reply_markup=_analytics_kb(period),
            )
        except TelegramBadRequest:
            pass
        return

    try:
        await callback.message.edit_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=_analytics_kb(period),
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            logger.warning("TelegramBadRequest в analytics: %s", e)


@user_router.message(Command("reindex"))
async def cmd_reindex(message: Message):
    if message.from_user.id not in settings.admin_ids_set:
        return

    await message.answer("🔄 Запущена пересборка индекса. Это может занять время...")
    try:
        await asyncio.to_thread(load_data_to_db, force=True)
        _RESULTS_CACHE.clear()
        await message.answer(
            "✅ Индекс успешно пересобран, кэш очищен.",
            reply_markup=kb_with_home([]),
        )
    except Exception as e:
        logger.exception("Ошибка при переиндексации")
        await message.answer(f"❌ Ошибка переиндексации: {e}")


# ───── Главное меню ─────
MAIN_MENU_TEXT = (
    "🏠 <b>Главное меню</b>\n\n"
    "🎯 <b>Подбор</b> — найти вузы по баллам ЕГЭ\n"
    "⭐ <b>Избранное</b> — сохранённые вузы\n"
    "🕐 <b>История</b> — последние поиски\n"
    "ℹ️ <b>Помощь</b> — как я работаю"
)


@user_router.message(Command("menu"))
async def cmd_menu(message: Message):
    if is_maintenance(message.from_user.id):
        await message.answer("🛠 Бот обновляет базу вузов, вернемся через 5 минут.")
        return
    await send_menu(
        message,
        message.from_user.id,
        MAIN_MENU_TEXT,
        build_main_menu(message.from_user.id),
    )


@user_router.callback_query(F.data == "open_menu")
async def cb_open_menu(callback: CallbackQuery):
    await callback.answer()
    try:
        await callback.message.edit_text(
            MAIN_MENU_TEXT,
            reply_markup=build_main_menu(callback.from_user.id),
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            logger.warning("TelegramBadRequest при открытии меню: %s", e)


# ───── Пользовательские команды ─────
@user_router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    if is_maintenance(message.from_user.id):
        await message.answer("🛠 Бот обновляет базу вузов, вернемся через 5 минут.")
        return
    await state.clear()
    name = html.escape(message.from_user.first_name or "друг")
    logger.info("Старт от user_id=%s", message.from_user.id)
    await message.answer(
        f"👋 Привет, {name}!\n\n"
        "Я — бот-помощник абитуриента. Помогу подобрать вузы по твоим баллам ЕГЭ "
        "и интересам.",
        parse_mode=ParseMode.HTML,
    )
    await send_menu(
        message,
        message.from_user.id,
        MAIN_MENU_TEXT,
        build_main_menu(message.from_user.id),
    )


@user_router.message(Command("help"))
async def cmd_help(message: Message):
    if is_maintenance(message.from_user.id):
        await message.answer("🛠 Бот обновляет базу вузов, вернемся через 5 минут.")
        return
    await send_menu(
        message,
        message.from_user.id,
        "📖 <b>Как я работаю</b>\n\n"
        "1. Спрашиваю город (или «Любой»).\n"
        "2. Спрашиваю направление — можно неточно, я использую нейросеть.\n"
        "3. Спрашиваю суммарный балл ЕГЭ.\n"
        "4. Показываю подходящие вузы.\n\n"
        "Команды:\n"
        "/menu — главное меню\n"
        "/task — начать подбор\n"
        "/favorites — избранные вузы\n"
        "/history — история поисков\n"
        "/stop — прервать текущий диалог\n"
        "«Отмена» — то же самое, что /stop.",
        kb_with_home([]),
    )


@user_router.callback_query(F.data == "open_help")
async def cb_open_help(callback: CallbackQuery):
    await callback.answer()
    try:
        await callback.message.edit_text(
            "📖 <b>Как я работаю</b>\n\n"
            "1. Спрашиваю город (или «Любой»).\n"
            "2. Спрашиваю направление — можно неточно, я использую нейросеть.\n"
            "3. Спрашиваю суммарный балл ЕГЭ.\n"
            "4. Показываю подходящие вузы.\n\n"
            "Нажми «🎯 Подбор» в главном меню, чтобы начать.",
            reply_markup=kb_with_home([]),
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            logger.warning("TelegramBadRequest в open_help: %s", e)


@user_router.message(Command("stop"))
async def cmd_stop(message: Message, state: FSMContext):
    current = await state.get_state()
    if current is None:
        await send_menu(
            message,
            message.from_user.id,
            "🤷 Сейчас нет активного диалога.",
            build_main_menu(message.from_user.id),
        )
        return
    await state.clear()
    await send_menu(
        message,
        message.from_user.id,
        "🛑 Диалог прерван.",
        build_main_menu(message.from_user.id),
    )


@user_router.message(F.text.casefold() == "отмена")
async def cmd_cancel(message: Message, state: FSMContext):
    current = await state.get_state()
    if current is None:
        await send_menu(
            message,
            message.from_user.id,
            "🤷 Сейчас нет активного диалога.",
            build_main_menu(message.from_user.id),
        )
        return
    await state.clear()
    await send_menu(
        message,
        message.from_user.id,
        "❌ Поиск отменён.",
        build_main_menu(message.from_user.id),
    )


# ───── Favorites & History ─────
@user_router.message(Command("favorites"))
async def cmd_favorites(message: Message):
    text, kb, _ = _build_favorites_view(message.from_user.id)
    if text is None:
        await send_menu(
            message,
            message.from_user.id,
            "⭐ У тебя пока нет избранных вузов.\n\n"
            "После поиска нажми ⭐ под результатами.",
            build_main_menu(message.from_user.id),
        )
        return
    await message.answer(
        text,
        reply_markup=kb,
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )


@user_router.callback_query(F.data == "open_favorites")
async def cb_open_favorites(callback: CallbackQuery):
    await callback.answer()
    text, kb, _ = _build_favorites_view(callback.from_user.id)
    if text is None:
        try:
            await callback.message.edit_text(
                "⭐ У тебя пока нет избранных вузов.\n\n"
                "После поиска нажми ⭐ под результатами.",
                reply_markup=kb_with_home([]),
            )
        except TelegramBadRequest:
            pass
        return
    try:
        await callback.message.edit_text(
            text,
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            logger.warning("TelegramBadRequest в open_favorites: %s", e)


@user_router.message(Command("history"))
async def cmd_history(message: Message):
    text, kb, _ = _build_history_view(message.from_user.id)
    if text is None:
        await send_menu(
            message,
            message.from_user.id,
            "🕐 История пуста — ты ещё не делал поисков.",
            build_main_menu(message.from_user.id),
        )
        return
    await message.answer(
        text,
        reply_markup=kb,
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )


@user_router.callback_query(F.data == "open_history")
async def cb_open_history(callback: CallbackQuery):
    await callback.answer()
    text, kb, _ = _build_history_view(callback.from_user.id)
    if text is None:
        try:
            await callback.message.edit_text(
                "🕐 История пуста — ты ещё не делал поисков.",
                reply_markup=kb_with_home([]),
            )
        except TelegramBadRequest:
            pass
        return
    try:
        await callback.message.edit_text(
            text,
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            logger.warning("TelegramBadRequest в open_history: %s", e)


# ───── Добавление в избранное ─────
@user_router.callback_query(F.data.startswith("fav:"))
async def cb_fav(callback: CallbackQuery):
    user_id = callback.from_user.id
    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer()
        return
    key = parts[1]
    try:
        idx = int(parts[2])
    except ValueError:
        await callback.answer()
        return

    data = _cache_get(key)
    if not data:
        await callback.answer("Список устарел, сделай поиск заново", show_alert=True)
        return

    _, _, _, metas = data
    if idx < 0 or idx >= len(metas):
        await callback.answer("Вуз не найден", show_alert=True)
        return

    m = metas[idx]
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO favorites "
                "(user_id, university, url, city, directions, budget, paid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    m.get("university", ""),
                    m.get("url", ""),
                    m.get("city", ""),
                    m.get("directions", ""),
                    m.get("budget", ""),
                    m.get("paid", ""),
                ),
            )
            added = cur.rowcount > 0
    except Exception:
        logger.exception("Ошибка записи в favorites")
        await callback.answer("Ошибка, попробуй позже", show_alert=True)
        return

    if added:
        await callback.answer("⭐ Добавлено в избранное")
    else:
        await callback.answer("Уже в избранном")


@user_router.callback_query(F.data.startswith("favdel:"))
async def cb_favdel(callback: CallbackQuery):
    user_id = callback.from_user.id
    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer()
        return
    try:
        fav_id = int(parts[1])
        page = int(parts[2])
    except ValueError:
        await callback.answer()
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "DELETE FROM favorites WHERE id = ? AND user_id = ?",
                (fav_id, user_id),
            )
            deleted = cur.rowcount > 0
    except Exception:
        logger.exception("Ошибка удаления из favorites")
        await callback.answer("Ошибка, попробуй позже", show_alert=True)
        return

    if not deleted:
        await callback.answer("Уже удалено")
        return

    await callback.answer("🗑 Удалено")

    text, kb, _ = _build_favorites_view(user_id, page)
    try:
        if text is None:
            await callback.message.edit_text(
                "⭐ Избранное пусто.\n\nПосле поиска нажми ⭐ под результатами.",
                reply_markup=kb_with_home([]),
            )
        else:
            await callback.message.edit_text(
                text,
                reply_markup=kb,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            logger.warning("TelegramBadRequest при обновлении избранного: %s", e)


@user_router.callback_query(F.data.startswith("favpage:"))
async def cb_favpage(callback: CallbackQuery):
    try:
        page = int(callback.data.split(":", 1)[1])
    except (ValueError, IndexError):
        await callback.answer()
        return
    await callback.answer()

    text, kb, _ = _build_favorites_view(callback.from_user.id, page)
    try:
        if text is None:
            await callback.message.edit_text(
                "⭐ Избранное пусто.",
                reply_markup=kb_with_home([]),
            )
        else:
            await callback.message.edit_text(
                text,
                reply_markup=kb,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            logger.warning("TelegramBadRequest в favpage: %s", e)


@user_router.callback_query(F.data.startswith("histpage:"))
async def cb_histpage(callback: CallbackQuery):
    try:
        page = int(callback.data.split(":", 1)[1])
    except (ValueError, IndexError):
        await callback.answer()
        return
    await callback.answer()

    text, kb, _ = _build_history_view(callback.from_user.id, page)
    try:
        if text is None:
            await callback.message.edit_text(
                "🕐 История пуста.",
                reply_markup=kb_with_home([]),
            )
        else:
            await callback.message.edit_text(
                text,
                reply_markup=kb,
                parse_mode=ParseMode.HTML,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
    except TelegramBadRequest as e:
        if "not modified" not in str(e).lower():
            logger.warning("TelegramBadRequest в histpage: %s", e)


# ───── Повтор поиска ─────
@user_router.callback_query(F.data.startswith("repeat:"))
async def cb_repeat(callback: CallbackQuery):
    user_id = callback.from_user.id
    try:
        qid = int(callback.data.split(":", 1)[1])
    except (ValueError, IndexError):
        await callback.answer()
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT city, specialty, score FROM queries "
                "WHERE id = ? AND user_id = ?",
                (qid, user_id),
            )
            row = cur.fetchone()
    except Exception:
        logger.exception("Ошибка чтения истории для повтора")
        await callback.answer("Ошибка, попробуй позже", show_alert=True)
        return

    if not row:
        await callback.answer("Запрос не найден", show_alert=True)
        return

    now = time.time()
    if now - _USER_LAST_SEARCH.get(user_id, 0.0) < SEARCH_COOLDOWN_SEC:
        await callback.answer("⏳ Подожди пару секунд", show_alert=True)
        return
    _USER_LAST_SEARCH[user_id] = now

    await callback.answer("🔎 Повторяю поиск...")

    try:
        await callback.message.delete()
    except TelegramBadRequest:
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass
    _LAST_MENU_MSG.pop(user_id, None)

    city = row["city"]
    specialty = row["specialty"]
    score = int(row["score"])

    try:
        all_metas = await asyncio.to_thread(search_vuz, city, specialty, score)
    except Exception:
        logger.exception("Ошибка при повторе поиска (user_id=%s)", user_id)
        await callback.message.answer("⚠️ Техническая ошибка при поиске. Попробуйте позже.")
        return

    city_disp = get_proper_city_name(city)
    spec_disp = str(specialty).capitalize()

    if not all_metas:
        header = f"🔎 Запрос: {html.escape(city_disp)} | {html.escape(spec_disp)} | {format_score(score)}\n\n"
        await callback.message.answer(
            header + "К сожалению, по вашим критериям ничего не найдено.",
            reply_markup=build_main_menu(user_id),
            link_preview_options=LinkPreviewOptions(is_disabled=True),
            parse_mode=ParseMode.HTML,
        )
        return

    key = _cache_put((city_disp, spec_disp, score), all_metas)
    await send_page(callback.message, key, 0, user_id)


# ───── Запуск подбора ─────
async def start_task_logic(message: Message, state: FSMContext, user_id: int):
    if is_maintenance(user_id):
        await message.answer("🛠 Бот обновляет базу вузов, вернемся через 5 минут.")
        return
    if not check_task_rate_limit(user_id):
        await send_menu(
            message,
            user_id,
            "⏳ Слишком много запросов. Подожди 10 секунд.",
            build_main_menu(user_id),
        )
        return

    await state.clear()
    await state.set_state(SearchCriteria.city)
    await message.answer(
        "🏙 Шаг 1/3. В каком городе планируешь учиться?\n"
        "Напиши название или нажми «Любой».",
        reply_markup=city_kb,
    )


@user_router.message(Command("task"))
async def cmd_task(message: Message, state: FSMContext):
    await start_task_logic(message, state, message.from_user.id)


@user_router.callback_query(F.data == "start_task")
async def cb_start_task(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    try:
        await callback.message.delete()
    except TelegramBadRequest:
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass
    _LAST_MENU_MSG.pop(callback.from_user.id, None)
    await start_task_logic(callback.message, state, callback.from_user.id)


# ───── Шаги FSM ─────
@user_router.message(SearchCriteria.city)
async def process_city(message: Message, state: FSMContext):
    if is_maintenance(message.from_user.id):
        await message.answer("🛠 Бот обновляет базу вузов, вернемся через 5 минут.")
        return
    text = (message.text or "").strip()
    if text.lower() != "любой" and not re.fullmatch(r"[A-Za-zА-Яа-яЁё\s\-]{2,40}", text):
        await message.answer(
            "❌ Название города — буквами (2–40 символов). Ещё раз:",
            reply_markup=city_kb,
        )
        return
    await state.update_data(city=text)
    await state.set_state(SearchCriteria.specialty)
    await message.answer(
        "📚 Шаг 2/3. Какое направление интересует?\n"
        "Например: Программная инженерия, IT, Лечебное дело.",
        reply_markup=cancel_kb,
    )


@user_router.message(SearchCriteria.specialty)
async def process_specialty(message: Message, state: FSMContext):
    if is_maintenance(message.from_user.id):
        await message.answer("🛠 Бот обновляет базу вузов, вернемся через 5 минут.")
        return
    text = (message.text or "").strip()
    text_lc = text.lower()

    if any(bad in text_lc for bad in _STOP_WORDS):
        antispam.register_explicit_violation(message.from_user.id)
        await message.answer(
            "❌ Пожалуйста, введи название направления без нецензурной лексики.",
            reply_markup=cancel_kb,
        )
        return

    if text_lc in _SKIP_SPECIALTY:
        await message.answer(
            "❌ Направление нужно указать конкретнее. Например: «Программная инженерия».\n"
            "Попробуй снова:",
            reply_markup=cancel_kb,
        )
        return

    is_cyrillic = re.fullmatch(r"[А-Яа-яЁё\s\-,]{3,80}", text)
    is_tech_term = text_lc in _EN_TECH_TERMS

    if not (is_cyrillic or is_tech_term):
        await message.answer(
            "❌ Название направления — русскими буквами (3–80 символов).\n"
            "Можно ещё ввести короткий IT-термин: IT, QA, AI, Python и т.п.\n"
            "Попробуй снова:",
            reply_markup=cancel_kb,
        )
        return

    await state.update_data(specialty=text)
    await state.set_state(SearchCriteria.score)
    await message.answer(
        "🎯 Шаг 3/3. Введи суммарный балл ЕГЭ (целое число от 110 до 310):",
        reply_markup=cancel_kb,
    )


@user_router.message(SearchCriteria.score)
async def process_score(message: Message, state: FSMContext):
    if is_maintenance(message.from_user.id):
        await message.answer("🛠 Бот обновляет базу вузов, вернемся через 5 минут.")
        return
    text = (message.text or "").strip()
    if not text.isdigit() or not (110 <= int(text) <= 310):
        await message.answer(
            "❌ Балл должен быть целым числом от 110 до 310. Попробуй снова:",
            reply_markup=cancel_kb,
        )
        return

    user_id = message.from_user.id
    now = time.time()

    if now - _USER_LAST_SEARCH.get(user_id, 0.0) < SEARCH_COOLDOWN_SEC:
        await send_menu(
            message,
            user_id,
            "⏳ Запросы на поиск слишком частые. Пожалуйста, подожди 5 секунд.",
            build_main_menu(user_id),
        )
        await state.clear()
        return

    _USER_LAST_SEARCH[user_id] = now
    _prune_rate_limit_caches(now)

    data = await state.get_data()
    city = data.get("city", "Любой")
    specialty = data.get("specialty", "")
    score = int(text)

    await state.clear()

    is_admin = 1 if user_id in settings.admin_ids_set else 0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO queries (user_id, city, specialty, score, is_admin) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, city, specialty, score, is_admin),
            )
    except Exception as e:
        logger.error("Ошибка записи аналитики: %s", e)

    loading = await message.answer(
        "🔎 Ищу подходящие варианты... ⏳",
        reply_markup=ReplyKeyboardRemove(),
    )

    logger.info("Поиск: user_id=%s city=%r spec=%r score=%d", user_id, city, specialty, score)

    try:
        all_metas = await asyncio.to_thread(search_vuz, city, specialty, score)
        await loading.delete()

        city_disp = get_proper_city_name(city)
        spec_disp = specialty.capitalize()

        if not all_metas:
            header = f"🔎 Запрос: {html.escape(city_disp)} | {html.escape(spec_disp)} | {format_score(score)}\n\n"
            await message.answer(
                header + "К сожалению, по вашим критериям ничего не найдено.",
                reply_markup=build_main_menu(user_id),
                link_preview_options=LinkPreviewOptions(is_disabled=True),
                parse_mode=ParseMode.HTML,
            )
            return

        key = _cache_put((city_disp, spec_disp, score), all_metas)
        await send_page(message, key, 0, user_id)

    except Exception:
        logger.exception("Ошибка при обработке поиска (user_id=%s)", user_id)
        try:
            await loading.delete()
        except Exception:
            pass
        await message.answer("⚠️ Техническая ошибка при поиске. Попробуйте позже.")


# ───── Пагинация ─────
@user_router.callback_query(F.data.startswith("page:"))
async def cb_page(callback: CallbackQuery):
    parts = callback.data.split(":")
    key = parts[1]
    page = int(parts[2])
    await callback.answer()
    await send_page(callback, key, page, callback.from_user.id)


@user_router.callback_query(F.data == "ignore")
async def cb_ignore(callback: CallbackQuery):
    await callback.answer()


# ───── Fallback ─────
@user_router.message(F.text)
async def fallback_text(message: Message, state: FSMContext):
    if is_maintenance(message.from_user.id):
        await message.answer("🛠 Бот обновляет базу вузов, вернемся через 5 минут.")
        return

    lowered = (message.text or "").lower()
    if antispam.should_notify(message.from_user.id):
        logger.info("Fallback от user_id=%s text=%r", message.from_user.id, lowered[:50])

    if any(w in lowered for w in ("привет", "здравств", "хай", "hello")):
        await send_menu(
            message,
            message.from_user.id,
            "👋 Привет! Я помогу подобрать вузы по баллам ЕГЭ.",
            build_main_menu(message.from_user.id),
        )
        return

    if any(w in lowered for w in ("спасибо", "благодар", "спс")):
        await message.answer("😊 Пожалуйста! Обращайся.")
        return

    await send_menu(
        message,
        message.from_user.id,
        "🤔 Не совсем понял.\n\n"
        "Я умею подбирать вузы по баллам ЕГЭ.",
        build_main_menu(message.from_user.id),
    )