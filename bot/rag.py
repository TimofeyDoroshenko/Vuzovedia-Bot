import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import difflib
import hashlib
import html
import importlib.util
import json
import logging
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

required_packages = {
    "numpy": "numpy",
    "sentence_transformers": "sentence-transformers",
    "dotenv": "python-dotenv",
}
for module_name, pip_name in required_packages.items():
    if importlib.util.find_spec(module_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH       = BASE_DIR / "vuz_data_full.json"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
METADATA_PATH   = BASE_DIR / "embeddings_metadata.pkl"
HASH_PATH       = BASE_DIR / ".embeddings_hash"

load_dotenv(BASE_DIR / ".env")

NULL_SCORE = 999
EMBEDDING_DIM = 384
ENCODE_BATCH_SIZE = 32

SEMANTIC_DISTANCE_LIMIT = 0.35
MIN_WORD_LEN = 5

LONG_PREFIX_LEN = 8
LONG_PREFIX_RATIO = 0.8
SHORT_PREFIX_MIN = 4
SHORT_PREFIX_RATIO = 0.9

FUZZY_MIN_LEN = 8
FUZZY_RATIO = 0.87

MAX_CITIES_IN_QUERY = 5

logger.info("Загружаю embedding-модель...")
_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
logger.info("Embedding-модель загружена.")

_embeddings = None
_metadatas = []

# Глобальные кэши для валидации адекватности запросов и поиска по коду ФГОС
_VALID_CITIES: dict[str, str] = {}
_GLOBAL_DIRECTIONS_TEXT: str = ""
_DIRECTION_NAMES: list[str] = []
_DIR_NAME_EMBEDDINGS: np.ndarray | None = None
_VALID_DIR_WORDS: set[str] = set()
_CODE_TO_NAME: dict[str, str] = {}

_GIBBERISH_RE = re.compile(
    r"[бвгджзйклмнпрстфхцчшщъь]{5,}|[аеёиоуыэюя]{4,}|(.)\1{2,}",
    re.IGNORECASE,
)

_CODE_INPUT_RE = re.compile(r"^(\d{2})[\.\-\s](\d{2})[\.\-\s](\d{2})$")
_CODE_EXTRACT_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{2})\b")

_ALLOWED_CONNECTORS = {"и", "в", "во", "на", "по", "с", "со", "для", "при", "за", "из", "от", "до", "или"}

# 1-й эшелон: флагманские вузы РФ.
_TIER1_PATTERNS = [
    re.compile(r"московский физико-технический|\bмфти\b"),
    re.compile(r"московский государственный университет имени м\.?\s*в\.?\s*ломоносова|\bмгу им"),
    re.compile(r"ядерный университет|\bмифи\b"),
    re.compile(r"высшая школа экономики|\bвшэ\b"),
    re.compile(r"баумана"),
    re.compile(r"\bмисис\b"),
    re.compile(r"\bитмо\b"),
    re.compile(r"^санкт-петербургский государственный университет$|\bспбгу\b"),
    re.compile(r"\bмгимо\b|институт международных отношений"),
    re.compile(r"сеченова|сеченовский"),
    re.compile(r"пирогова"),
    re.compile(r"кутафина|\bмгюа\b"),
]

# 2-й эшелон: ведущие политехнические, национальные исследовательские (НИУ) и федеральные вузы.
_TIER2_PATTERNS = [
    re.compile(r"санкт-петербургский политехнический|петра великого|\bспбпу\b"),
    re.compile(r"\bлэти\b|электротехнический университет «лэти»"),
    re.compile(r"\bмирэа\b|российский технологический университет"),
    re.compile(r"московский авиационный институт|\bмаи\b"),
    re.compile(r"московский энергетический институт|\bмэи\b"),
    re.compile(r"финансовый университет при правительстве"),
    re.compile(r"народного хозяйства и государственной службы|\bранхигс\b"),
    re.compile(r"экономический университет имени г\.?\s*в\.?\s*плеханова|плеханова"),
    re.compile(r"нефти и газа.*губкина|губкина"),
    re.compile(r"национальный исследовательский московский государственный строительный|\bниу мгсу\b"),
    re.compile(r"химико-технологический.*менделеева|\bрхту\b"),
    re.compile(r"дружбы народов|\bрудн\b"),
    re.compile(r"санкт-петербургский горный университет|горный университет императрицы"),
    re.compile(r"казанский \(приволжский\) федеральный|\bкфу\b"),
    re.compile(r"уральский федеральный|\bурфу\b"),
    re.compile(r"новосибирский национальный исследовательский государственный университет|\bнгу\b"),
    re.compile(r"нижегородский государственный университет.*лобачевского|лобачевского"),
    re.compile(r"томский политехнический|\bтпу\b"),
    re.compile(r"томский государственный университет$|\bниу тгу\b"),
    re.compile(r"южный федеральный университет|\bюфу\b"),
    re.compile(r"дальневосточный федеральный|\bдвфу\b"),
]

_NARROW_FLAGSHIP_RE = re.compile(r"сеченова|сеченовский|пирогова|кутафина|\bмгюа\b|\bмгимо\b")

CITY_ALIASES = {
    "мск": "Москва", "масква": "Москва", "msk": "Москва",
    "спб": "Санкт-Петербург", "спбг": "Санкт-Петербург", "питер": "Санкт-Петербург",
    "петербург": "Санкт-Петербург", "ленинград": "Санкт-Петербург",
    "spb": "Санкт-Петербург",
    "новосиб": "Новосибирск", "нск": "Новосибирск", "nsk": "Новосибирск",
    "екб": "Екатеринбург", "ебург": "Екатеринбург", "ekb": "Екатеринбург",
    "нн": "Нижний Новгород", "нижний": "Нижний Новгород", "nn": "Нижний Новгород",
    "ростов": "Ростов-на-Дону", "рнд": "Ростов-на-Дону", "ростовнадону": "Ростов-на-Дону",
    "rnd": "Ростов-на-Дону",
    "челяба": "Челябинск", "chel": "Челябинск",
    "казан": "Казань", "kzn": "Казань",
    "челны": "Набережные Челны", "набережныечелны": "Набережные Челны",
    "магнитка": "Магнитогорск",
    "королев": "Королёв",
}

SPECIALTY_SYNONYMS = {
    # ─── Русские синонимы ───
    "дома": "строительство",
    "здания": "строительство",
    "мосты": "строительство",
    "дороги": "строительство",
    "проектировать": ["строительство", "архитектура", "конструкторско"],
    "чертежи": ["строительство", "архитектура"],
    "архитектор": "архитектура",
    "игры": ["программная", "информатика"],
    "геймдев": ["программная", "информатика"],
    "сайты": ["программная", "информатика"],
    "приложения": ["программная", "информатика"],
    "ботов": ["программная", "информатика"],
    "нейросети": ["информатика", "математика"],
    "роботов": ["робототехника", "мехатроника"],
    "роботы": ["робототехника", "мехатроника"],
    "самолеты": "авиастроение",
    "ракеты": ["ракетостроение", "авиастроение"],
    "машины": "машиностроение",
    "автомобили": "машиностроение",
    "банки": ["экономика", "финансы"],
    "деньги": "экономика",
    "бизнес": ["менеджмент", "экономика"],
    "кино": "режиссура",
    "фильмы": "режиссура",
    "актер": "актерское",
    "актриса": "актерское",
    "петь": "вокальное",
    "музыка": "музыкальное",
    "животных": "ветеринария",
    "ветеринар": "ветеринария",
    "зубы": "стоматология",
    "детей": ["педагогическое", "педиатрия"],
    "полиция": ["юриспруденция", "правоохранительная"],
    "следователь": ["юриспруденция", "правоохранительная"],
    "лечить": "лечебное",
    "лечение": "лечебное",
    "лечу": "лечебное",
    "врач": "лечебное",
    "медик": "лечебное",
    "медицина": "лечебное",
    "медицинский": "лечебное",
    "айти": ["информатика", "программная", "компьютерные", "информационные"],
    "ит": ["информатика", "программная", "компьютерные", "информационные"],
    "программирование": ["программная", "информатика", "компьютерные"],
    "программировать": ["программная", "информатика", "компьютерные"],
    "программист": ["программная", "информатика", "компьютерные"],
    "разработчик": ["программная", "информатика", "компьютерные"],
    "кодер": ["программная", "информатика", "компьютерные"],
    "база": ["информационные", "информатика"],
    "базы": ["информационные", "информатика"],
    "базу": ["информационные", "информатика"],
    "данных": ["информационные", "информатика"],
    "рисование": "дизайн",
    "рисовать": "дизайн",
    "рисую": "дизайн",
    "художник": "живопись",
    "учитель": "педагогическое",
    "преподаватель": "педагогическое",
    "экономист": "экономика",
    "юрист": "юриспруденция",
    "адвокат": "юриспруденция",
    "инженер": "инженерное",
    "строитель": "строительство",
    "строить": "строительство",
    # Медицина
    "хирургия": "лечебное",
    "хирург": "лечебное",
    "хирургический": "лечебное",
    "терапия": "лечебное",
    "терапевт": "лечебное",
    "стоматолог": "стоматология",
    "педиатр": "педиатрия",
    "врача": "лечебное",
    "врачу": "лечебное",
    "врачом": "лечебное",
    "врачи": "лечебное",
    # Юриспруденция
    "юридический": "юриспруденция",
    "юридическая": "юриспруденция",
    "юридическое": "юриспруденция",
    "адвокатура": "юриспруденция",
    "прокуратура": "юриспруденция",
    "прокурор": "юриспруденция",
    "судья": "юриспруденция",
    "нотариус": "юриспруденция",
    # Педагогика
    "воспитатель": "педагогическое",
    "воспитание": "педагогическое",
    "педагог": "педагогическое",
    # Нефтегаз
    "бурение": "нефтегаз",
    "нефтедобыча": "нефтегаз",
    "нефтяник": "нефтегаз",
    "нефтяной": "нефтегаз",
    # Творчество
    "режиссер": "режиссура",
    "режиссёр": "режиссура",
    # Транспорт
    "логистика": "транспорт",
    "логист": "транспорт",
    # Маркетинг и падежи экономики
    "маркетинг": "менеджмент",
    "маркетолог": "менеджмент",
    "экономику": "экономика",
    "экономики": "экономика",
    "экономике": "экономика",
    "экономикой": "экономика",
    # Опечатки
    "програмирование": ["программная", "информатика", "компьютерные"],
    # ─── IT-термины на латинице ───
    "it": ["информатика", "программная", "компьютерные", "информационные"],
    "qa": ["программная", "информатика"],
    "ai": ["информатика", "математика", "компьютерные"],
    "ml": ["информатика", "математика", "компьютерные"],
    "web": ["программная", "информатика"],
    "data": ["информатика", "математика"],
    "sql": ["информатика", "информационные"],
    "python": ["программная", "информатика", "компьютерные"],
    "java": ["программная", "информатика", "компьютерные"],
    "javascript": ["программная", "информатика"],
    "kotlin": ["программная", "информатика"],
    "swift": ["программная", "информатика"],
    "golang": ["программная", "информатика"],
    "go": ["программная", "информатика"],
    "frontend": ["программная", "информатика"],
    "backend": ["программная", "информатика"],
    "fullstack": ["программная", "информатика"],
    "devops": ["информатика", "информационные"],
    "ux": "дизайн",
    "ui": "дизайн",
    "c++": ["программная", "информатика", "компьютерные"],
    "c#": ["программная", "информатика"],
    "data science": ["информатика", "математика"],
    "data-science": ["информатика", "математика"],
    "datascience": ["информатика", "математика"],
    "datascientist": ["информатика", "математика"],
    "датасаенс": ["информатика", "математика"],
}

CANONICAL_SYNONYM_TITLES = {
    "дома": "Строительство",
    "здания": "Строительство",
    "мосты": "Строительство",
    "дороги": "Строительство",
    "архитектор": "Архитектура",
    "игры": "Программная инженерия",
    "геймдев": "Программная инженерия",
    "сайты": "Программная инженерия",
    "приложения": "Программная инженерия",
    "ботов": "Программная инженерия",
    "нейросети": "Прикладная математика и информатика",
    "роботов": "Мехатроника и робототехника",
    "роботы": "Мехатроника и робототехника",
    "машины": "Машиностроение",
    "автомобили": "Машиностроение",
    "деньги": "Экономика",
    "бизнес": "Менеджмент",
    "кино": "Режиссура кино и телевидения",
    "фильмы": "Режиссура кино и телевидения",
    "животных": "Ветеринария",
    "ветеринар": "Ветеринария",
    "зубы": "Стоматология",
    "лечить": "Лечебное дело",
    "лечение": "Лечебное дело",
    "лечу": "Лечебное дело",
    "врач": "Лечебное дело",
    "врача": "Лечебное дело",
    "врачу": "Лечебное дело",
    "врачом": "Лечебное дело",
    "врачи": "Лечебное дело",
    "медик": "Лечебное дело",
    "медицина": "Лечебное дело",
    "медицинский": "Лечебное дело",
    "терапия": "Лечебное дело",
    "терапевт": "Лечебное дело",
    "хирург": "Лечебное дело",
    "хирургия": "Лечебное дело",
    "хирургический": "Лечебное дело",
    "стоматолог": "Стоматология",
    "педиатр": "Педиатрия",
    "айти": "Айти",
    "ит": "Айти",
    "it": "Айти",
    "программирование": "Программирование",
    "програмирование": "Программирование",
    "программировать": "Программирование",
    "программист": "Программирование",
    "разработчик": "Программирование",
    "кодер": "Программирование",
    "рисование": "Дизайн",
    "рисовать": "Дизайн",
    "рисую": "Дизайн",
    "ux": "Дизайн (UX/UI)",
    "ui": "Дизайн (UX/UI)",
    "художник": "Живопись",
    "учитель": "Педагогическое образование",
    "преподаватель": "Педагогическое образование",
    "педагог": "Педагогическое образование",
    "воспитатель": "Педагогическое образование",
    "воспитание": "Педагогическое образование",
    "экономист": "Экономика",
    "экономику": "Экономика",
    "экономики": "Экономика",
    "экономике": "Экономика",
    "экономикой": "Экономика",
    "юрист": "Юриспруденция",
    "юридический": "Юриспруденция",
    "юридическая": "Юриспруденция",
    "юридическое": "Юриспруденция",
    "адвокат": "Юриспруденция",
    "адвокатура": "Юриспруденция",
    "прокурор": "Юриспруденция",
    "прокуратура": "Юриспруденция",
    "судья": "Юриспруденция",
    "нотариус": "Юриспруденция",
    "строитель": "Строительство",
    "строить": "Строительство",
    "бурение": "Нефтегазовое дело",
    "нефтедобыча": "Нефтегазовое дело",
    "нефтяник": "Нефтегазовое дело",
    "нефтяной": "Нефтегазовое дело",
    "режиссер": "Режиссура",
    "режиссёр": "Режиссура",
    "база данных": "Информационные системы и БД",
    "базы данных": "Информационные системы и БД",
    "базу данных": "Информационные системы и БД",
}

# Лаконичные сокращения для сверхдлинных названий ФГОС, чтобы каждое направление
# помещалось строго в одну строку и не переносилось в Telegram
_COMPACT_DIR_REPLACEMENTS = (
    ("Информационно-аналитические системы безопасности", "Инф.-аналитические системы безопасности"),
    ("Информационная безопасность телекоммуникационных систем", "Инф. безопасность телеком-систем"),
    ("Информационная безопасность автоматизированных систем", "Инф. безопасность автомат. систем"),
    ("Инфокоммуникационные технологии и системы связи", "Инфокоммуникации и системы связи"),
    ("Информатика и вычислительная техника", "Информатика и ВТ"),
    ("Фундаментальная информатика и информационные технологии", "Фундаментальная информатика и ИТ"),
    ("Математическое обеспечение и администрирование информационных систем", "Мат. обеспечение и админ. ИС"),
    ("Применение и эксплуатация автоматизированных систем специального назначения", "Автоматизированные системы спецназначения"),
    ("Автоматизация технологических процессов и производств", "Автоматизация техпроцессов и производств"),
    ("Конструирование и технология электронных средств", "Конструирование электронных средств"),
    ("Государственное и муниципальное управление", "Гос. и муниципальное управление"),
)


def _compact_dir_name(name: str, max_len: int = 40) -> str:
    """Делает слишком длинные названия направлений компактными, чтобы они всегда
    умещались на одной строке на одном уровне.
    """
    res = name.strip()
    for old, new in _COMPACT_DIR_REPLACEMENTS:
        if old.lower() in res.lower():
            res = re.sub(re.escape(old), new, res, flags=re.IGNORECASE)
    if len(res) > max_len:
        res = res[: max_len - 1].rstrip() + "…"
    return res


def normalize_text(text: str) -> str:
    return text.lower().replace("-", "").replace(" ", "").strip()


def _format_city_title(raw: str) -> str:
    t = raw.strip().title()
    return re.sub(r"-На-", "-на-", t)


def canonicalize_city(name: str) -> str:
    key = normalize_text(name)
    return CITY_ALIASES.get(key, name.strip())


def _resolve_single_city(raw_city: str, allow_fuzzy: bool = True) -> tuple[bool, str, list[str]]:
    cleaned = raw_city.strip()
    if not cleaned:
        return False, "", []
    if cleaned.lower() in ("любой", "любой город", "все"):
        return True, "Любой", []

    canonical = canonicalize_city(cleaned)
    norm = normalize_text(canonical)

    if not _VALID_CITIES:
        return True, _format_city_title(canonical), []

    if norm in _VALID_CITIES:
        return True, _format_city_title(_VALID_CITIES[norm]), []

    if allow_fuzzy:
        close_exact = difflib.get_close_matches(norm, _VALID_CITIES.keys(), n=1, cutoff=0.82)
        if close_exact:
            return True, _format_city_title(_VALID_CITIES[close_exact[0]]), []

    close_hints = difflib.get_close_matches(norm, _VALID_CITIES.keys(), n=3, cutoff=0.6)
    suggestions = [_format_city_title(_VALID_CITIES[k]) for k in close_hints]
    return False, cleaned, suggestions


def validate_and_resolve_city(raw_city: str, allow_fuzzy: bool = True) -> tuple[bool, str, list[str]]:
    cleaned = raw_city.strip()
    if cleaned.lower() in ("любой", "любой город", "все"):
        return True, "Любой", []

    parts = [p.strip() for p in re.split(r"[,;]+", cleaned) if p.strip()]
    if not parts:
        return False, raw_city, []

    resolved_cities: list[str] = []
    seen_norms: set[str] = set()

    for part in parts:
        if part.lower() in ("любой", "любой город", "все"):
            return True, "Любой", []

        ok, canon, suggestions = _resolve_single_city(part, allow_fuzzy=allow_fuzzy)
        if not ok:
            return False, part, suggestions

        c_norm = normalize_text(canon)
        if c_norm not in seen_norms:
            seen_norms.add(c_norm)
            resolved_cities.append(canon)

    if len(resolved_cities) > MAX_CITIES_IN_QUERY:
        resolved_cities = resolved_cities[:MAX_CITIES_IN_QUERY]

    return True, ", ".join(resolved_cities), []


def get_proper_city_name(city: str) -> str:
    if city.lower() in ("любой", "любой город"):
        return "Любой город"
    ok, canon, _ = validate_and_resolve_city(city, allow_fuzzy=True)
    if ok:
        return "Любой город" if canon == "Любой" else canon
    return _format_city_title(city)


def _is_undergrad_direction(d: dict) -> bool:
    code = (d.get("code") or "").strip()
    if code:
        parts = code.split(".")
        if len(parts) == 3 and parts[1] not in ("03", "05"):
            return False

    level = (d.get("level") or "").lower()
    if any(bad in level for bad in ("магистрат", "аспирант", "ординат", "ассистент", "адъюнкт")):
        return False

    return True


def _extract_codes_map(item: dict) -> dict[str, str]:
    dl = item.get("directions_list") or []
    codes_map: dict[str, str] = {}
    for d in dl:
        if not _is_undergrad_direction(d):
            continue
        code = (d.get("code") or "").strip()
        name = (d.get("name") or "").strip()
        if code and name and code not in codes_map:
            codes_map[code] = name
    return codes_map


def _code_level_priority(code: str) -> tuple[int, str]:
    parts = code.split(".")
    if len(parts) == 3:
        if parts[1] == "03":
            return (0, code)
        if parts[1] == "05":
            return (1, code)
    return (2, code)


def _extract_dir_items(item: dict) -> list[dict]:
    dl = item.get("directions_list") or []
    items: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for d in dl:
        if not _is_undergrad_direction(d):
            continue
        name = (d.get("name") or "").strip()
        if not name:
            continue
        code = (d.get("code") or "").strip()
        url = (d.get("url") or "").strip()
        key = (code, name.lower())
        if key in seen:
            continue
        seen.add(key)
        items.append({"name": name, "code": code, "url": url})

    items.sort(key=lambda x: _code_level_priority(x.get("code", "")))
    return items


def _directions_str(item: dict) -> str:
    dl = item.get("directions_list") or []
    names, seen = [], set()
    for d in dl:
        if not _is_undergrad_direction(d):
            continue
        name = (d.get("name") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    if names:
        return ", ".join(names)
    if dl:
        return "Не указано"
    return str(item.get("directions") or "Не указано")


def _enrich_metas_if_needed() -> None:
    if not _metadatas or not JSON_PATH.exists():
        return
    if "dir_items_v3" in _metadatas[0] and "codes_map" in _metadatas[0]:
        return

    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception:
        logger.warning("Не удалось прочитать JSON для фильтрации магистратуры/аспирантуры")
        return

    url_to_info: dict[str, tuple[dict, list, str]] = {}
    univ_to_info: dict[str, tuple[dict, list, str]] = {}
    for item in raw_data:
        cmap = _extract_codes_map(item)
        ditems = _extract_dir_items(item)
        dstr = _directions_str(item)
        u_url = str(item.get("url") or "")
        u_name = str(item.get("university") or "")
        if u_url:
            url_to_info[u_url] = (cmap, ditems, dstr)
        if u_name:
            univ_to_info[u_name] = (cmap, ditems, dstr)

    for m in _metadatas:
        info = url_to_info.get(m.get("url", "")) or univ_to_info.get(m.get("university", ""))
        if info:
            m["codes_map"] = info[0]
            m["dir_items_v3"] = info[1]
            m["directions"] = info[2]
        else:
            m.setdefault("codes_map", {})
            m.setdefault("dir_items_v3", [])
        m.pop("dir_items", None)
        m.pop("dir_items_v2", None)

    _save_index()
    logger.info("Метаданные очищены от магистратуры и аспирантуры (v3).")


def _rebuild_validation_caches() -> None:
    global _VALID_CITIES, _GLOBAL_DIRECTIONS_TEXT, _DIRECTION_NAMES, _VALID_DIR_WORDS, _CODE_TO_NAME
    _enrich_metas_if_needed()

    cities: dict[str, str] = {}
    unique_dirs: set[str] = set()
    dir_words: set[str] = set()
    code_to_name: dict[str, str] = {}

    for m in _metadatas:
        c_norm = m.get("city_norm")
        c_name = m.get("city")
        if c_norm and c_name and c_norm != normalize_text("Не указан"):
            cities[c_norm] = _format_city_title(c_name)

        for raw_dir in str(m.get("directions", "")).split(","):
            d_clean = raw_dir.strip()
            if d_clean and d_clean.lower() != "не указано":
                unique_dirs.add(d_clean)
                for w in re.findall(r"[а-яёa-z0-9]+", d_clean.lower()):
                    if len(w) >= 2 and w not in _ALLOWED_CONNECTORS:
                        dir_words.add(w)

        for code, d_name in (m.get("codes_map") or {}).items():
            if code not in code_to_name:
                code_to_name[code] = d_name
        global _DIR_NAME_EMBEDDINGS
        if _DIRECTION_NAMES:
            _DIR_NAME_EMBEDDINGS = _encode([f"passage: {d}" for d in _DIRECTION_NAMES])

    _VALID_CITIES = cities
    _DIRECTION_NAMES = sorted(unique_dirs)
    _GLOBAL_DIRECTIONS_TEXT = " , ".join(_DIRECTION_NAMES).lower()
    _VALID_DIR_WORDS = dir_words
    _CODE_TO_NAME = code_to_name
    logger.info(
        "Кэш валидации обновлён: %d городов, %d направлений (03/05), %d слов, %d кодов ФГОС",
        len(_VALID_CITIES),
        len(_DIRECTION_NAMES),
        len(_VALID_DIR_WORDS),
        len(_CODE_TO_NAME),
    )


def _is_word_in_vocabulary(word: str) -> bool:
    if word in SPECIALTY_SYNONYMS or word in _VALID_DIR_WORDS:
        return True

    if len(word) < MIN_WORD_LEN:
        if len(word) >= 3:
            for dw in _VALID_DIR_WORDS:
                if abs(len(word) - len(dw)) <= 1 and _common_prefix_len(word, dw) >= 3:
                    return True
        return False

    for dw in _VALID_DIR_WORDS:
        if len(dw) < MIN_WORD_LEN:
            continue
        if abs(len(word) - len(dw)) > 3:
            continue

        prefix = _common_prefix_len(word, dw)
        min_len = min(len(word), len(dw))
        max_len = max(len(word), len(dw))

        if prefix == max_len:
            return True
        if prefix == min_len and len(word) < len(dw):
            return True
        if prefix >= LONG_PREFIX_LEN and prefix / min_len >= LONG_PREFIX_RATIO:
            return True
        if prefix >= SHORT_PREFIX_MIN and prefix / min_len >= SHORT_PREFIX_RATIO:
            return True
        if (
            len(word) >= FUZZY_MIN_LEN
            and len(dw) >= FUZZY_MIN_LEN
            and difflib.SequenceMatcher(None, word, dw).ratio() >= FUZZY_RATIO
        ):
            return True

    return False


def _word_matches_direction(word: str, direction_lc: str) -> bool:
    syn = SPECIALTY_SYNONYMS.get(word)
    candidates = [word]
    if syn:
        candidates.extend([syn] if isinstance(syn, str) else syn)

    for cand in candidates:
        if _whole_word_match(cand, direction_lc) or _lexical_overlap(cand, direction_lc):
            return True
        if 3 <= len(cand) < MIN_WORD_LEN:
            for dw in re.findall(r"[а-яёa-z0-9]+", direction_lc):
                if abs(len(cand) - len(dw)) <= 1 and _common_prefix_len(cand, dw) >= 3:
                    return True
    return False

# Разговорные слова-связки, которые мы разрешаем в естественной речи,
# но очищаем перед проверкой сути запроса
# Разговорные слова-связки, местоимения и общие глаголы, которые мы разрешаем
# в естественной речи, но очищаем перед проверкой сути запроса
_NATURAL_FILLERS = {
    # Местоимения и частицы
    "я", "мне", "меня", "мной", "мы", "нам", "ты", "вы", "он", "она", "они",
    "свой", "свои", "свою", "свое", "это", "этом", "бы", "же", "ли", "вот",
    "как", "где", "куда", "кто", "что", "чем", "какой", "какое", "какие",
    # Глаголы желания и действия
    "хочу", "хотел", "хотела", "хочется", "буду", "могу", "умею", "знаю",
    "люблю", "нравится", "мечтаю", "планирую", "интересно", "интересует",
    "быть", "стать", "работать", "учиться", "пойти", "поступить", "поступать",
    "заниматься", "делать", "создавать", "писать", "разрабатывать", "придумывать",
    "изучать", "учить", "обучать", "помогать", "спасать", "открыть", "вести",
    # Общие дополнения и наречия
    "людей", "людям", "человека", "людьми", "компании", "жизни", "мире",
    "профессионально", "круто", "хорошо", "очень", "просто", "вообще", "реально",
    "сфера", "сфере", "область", "области", "направлению",
    "дело", "работа", "профессия", "специальность", "направление", "факультет",
    "дом", "дома", "здания", "зданий", "сооружения",
}


# Многословные разговорные фразы, которые сразу маппятся в точное направление
_PHRASE_SYNONYMS = {
    ("лечить", "животных"): "Ветеринария",
    ("лечение", "животных"): "Ветеринария",
    ("врач", "животных"): "Ветеринария",
    ("лечить", "зубы"): "Стоматология",
    ("лечение", "зубов"): "Стоматология",
    ("лечить", "детей"): "Педиатрия",
    ("детский", "врач"): "Педиатрия",
    ("учить", "детей"): "Педагогическое образование",
    ("обучать", "детей"): "Педагогическое образование",
    ("строить", "дома"): "Строительство",
    ("строить", "здания"): "Строительство",
    ("база", "данных"): "Информационные системы и БД",
    ("базы", "данных"): "Информационные системы и БД",
    ("базу", "данных"): "Информационные системы и БД",
}

# Широкие слова-глаголы, которые при наличии второго конкретного слова ("животных", "зубы")
# не должны конфликтовать с ним при поиске в ФГОС
_BROAD_ACTION_WORDS = {"лечить", "лечение", "лечу", "врач", "строить", "учить", "инженер"}


def _resolve_natural_language_to_canonical(cleaned_query: str, content_words: list[str]) -> tuple[bool, str]:
    """Превращает естественный запрос ('я хочу лечить животных', 'строить дома', 'создавать сайты')
    в каноническое направление из базы, отсекая мусор и троллинг.
    """
    global _DIR_NAME_EMBEDDINGS

    # 1. Проверяем, все ли слова знакомы базе ФГОС или словарю синонимов
    known_words = [w for w in content_words if _is_word_in_vocabulary(w)]
    unknown_words = [w for w in content_words if w not in known_words]

    if unknown_words:
        return False, f"Слово «{unknown_words[0]}» не похоже на учебное направление."

    if not known_words:
        return False, "Опиши направление чуть конкретнее (например: «лечить людей», «программирование», «дизайн»)."

    # 2. Проверяем устойчивые словосочетания ("лечить животных" -> "Ветеринария", "лечить зубы" -> "Стоматология")
    kw_set = set(known_words)
    for phrase_tuple, target_title in _PHRASE_SYNONYMS.items():
        if set(phrase_tuple).issubset(kw_set):
            return True, target_title

    # 3. Если в запросе есть широкий глагол ("лечить") + более узкое слово ("животных" / "стоматолог"),
    # оставляем узкое слово главным, чтобы "лечебное" и "ветеринария" не требовались одновременно в одной строке
    if len(known_words) > 1:
        narrow_words = [w for w in known_words if w not in _BROAD_ACTION_WORDS]
        if narrow_words:
            known_words = narrow_words

    # 4. Если осталось 1 ключевое слово из словаря канонических названий
    if len(known_words) == 1 and known_words[0] in CANONICAL_SYNONYM_TITLES:
        return True, CANONICAL_SYNONYM_TITLES[known_words[0]]

    # 5. Проверяем совпадение всех ключевых слов с реальными направлениями базы
    if _DIRECTION_NAMES:
        cohesive_dirs = [
            d for d in _DIRECTION_NAMES
            if all(_word_matches_direction(w, d.lower()) for w in known_words)
        ]
        if cohesive_dirs:
            for d in cohesive_dirs:
                if normalize_text(d) == normalize_text(" ".join(known_words)):
                    return True, d
            cohesive_dirs.sort(key=len)
            return True, cohesive_dirs[0]

    # 6. Если все слова из нашего словаря синонимов (например, "создавать игры и сайты"),
    # берём каноническое название по первому распознанному синониму
    for w in known_words:
        if w in CANONICAL_SYNONYM_TITLES:
            return True, CANONICAL_SYNONYM_TITLES[w]

    # 7. Нейросетевой фоллбэк (E5): лениво строим эмбеддинги направлений, если они ещё не созданы
    if _DIRECTION_NAMES:
        if _DIR_NAME_EMBEDDINGS is None:
            _DIR_NAME_EMBEDDINGS = _encode([f"passage: {d}" for d in _DIRECTION_NAMES])
        expanded = _expand_query(" ".join(known_words))
        q_vec = _encode([f"query: {expanded}"])[0]
        sims = _DIR_NAME_EMBEDDINGS @ q_vec
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= 0.76:
            return True, _DIRECTION_NAMES[best_idx]

    return False, "Не удалось подобрать направление. Попробуй написать проще (например: «Лечебное дело» или «Информатика»)."

def validate_and_resolve_specialty(raw_spec: str) -> tuple[bool, str, str]:
    cleaned = " ".join(raw_spec.strip().split())
    spec_lc = cleaned.lower()

    # 1. Проверка на код направления (например: 09.03.01, 09-03-01, 09 03 01)
    code_match = _CODE_INPUT_RE.match(spec_lc)
    if code_match:
        g1, g2, g3 = code_match.groups()
        norm_code = f"{g1}.{g2}.{g3}"

        if g2 == "04":
            alt_code = f"{g1}.03.{g3}"
            alt_hint = f" (например: {alt_code} — {_CODE_TO_NAME[alt_code]})" if alt_code in _CODE_TO_NAME else ""
            return (
                False,
                "",
                f"Код «{norm_code}» относится к магистратуре, куда не поступают по ЕГЭ. "
                f"Используй код бакалавриата XX.03.XX или специалитета XX.05.XX{alt_hint}.",
            )
        if g2 in ("06", "07", "08", "09"):
            return (
                False,
                "",
                f"Код «{norm_code}» относится к аспирантуре/ординатуре. "
                "Используй код бакалавриата (XX.03.XX) или специалитета (XX.05.XX).",
            )

        if norm_code in _CODE_TO_NAME:
            dir_name = _CODE_TO_NAME[norm_code]
            return True, f"{norm_code} — {dir_name}", ""

        if g2 == "00" and g3 == "00":
            prefix = f"{g1}."
            group_matches = [name for c, name in _CODE_TO_NAME.items() if c.startswith(prefix)]
            if group_matches:
                return True, f"{norm_code} (группа направлений)", ""

        close_codes = difflib.get_close_matches(norm_code, _CODE_TO_NAME.keys(), n=3, cutoff=0.75)
        hint = ""
        if close_codes:
            examples = [f"{c} ({_CODE_TO_NAME[c]})" for c in close_codes]
            hint = "\nВозможно, ты искал: " + ", ".join(examples)
        return (
            False,
            "",
            f"Код направления «{norm_code}» не найден в базе бакалавриата и специалитета.{hint}",
        )

    if _CODE_EXTRACT_RE.search(spec_lc):
        return True, cleaned, ""

    # 2. Проверка на случайный набор символов («ааааа», «фывапр»)
    if _GIBBERISH_RE.search(spec_lc):
        return False, "", "Похоже на случайный набор букв."

    if spec_lc not in SPECIALTY_SYNONYMS and not re.search(r"[аеёиоуыэюяaeiouy]", spec_lc):
        return False, "", "Не удалось распознать слова в названии направления."

    # 3. Прямое совпадение со словарем синонимов или канонических названий
    if spec_lc in SPECIALTY_SYNONYMS:
        if spec_lc in CANONICAL_SYNONYM_TITLES:
            return True, CANONICAL_SYNONYM_TITLES[spec_lc], ""
        if re.fullmatch(r"[a-z+#\s\-]+", spec_lc):
            norm_disp = spec_lc.upper() if len(spec_lc) <= 4 else spec_lc.title()
        else:
            norm_disp = spec_lc.capitalize()
        return True, norm_disp, ""

    if spec_lc in CANONICAL_SYNONYM_TITLES:
        return True, CANONICAL_SYNONYM_TITLES[spec_lc], ""

    # 4. Обработка естественного языка и пословная проверка на мусор («рака яиц» и т.д.)
    tokens = re.findall(r"[а-яёa-z0-9+#]+", spec_lc)
    content_words = [
        w for w in tokens
        if w not in _ALLOWED_CONNECTORS and w not in _NATURAL_FILLERS
    ]

    # Если все слова попали в филлеры (например, юзер написал просто "дома" или "людей"),
    # пробуем взять из токенов те слова, которые есть в нашем словаре синонимов
    if not content_words:
        content_words = [w for w in tokens if w in SPECIALTY_SYNONYMS or w in CANONICAL_SYNONYM_TITLES]

    if not content_words:
        return False, "", "Направление нужно указать конкретнее (например: «лечить людей», «строить дома», «09.03.01»)."

    if len(content_words) > 6:
        return False, "", "Слишком длинный запрос. Опиши направление в 1–4 словах."

    # Проверяем, что каждое значимое слово есть в словаре ФГОС или синонимов
    if _VALID_DIR_WORDS:
        for w in content_words:
            if not _is_word_in_vocabulary(w):
                return (
                    False,
                    "",
                    f"Слово «{w}» не относится к названиям вузовских направлений. "
                    "Введи реальное направление (например: «Лечебное дело», «Информатика», «09.03.01»).",
                )

    # 5. Превращаем естественную фразу в каноническое название через словарь или нейросеть (E5)
    ok, result_or_err = _resolve_natural_language_to_canonical(cleaned, content_words)
    if not ok:
        return False, "", result_or_err

    return True, result_or_err, ""


def _safe_int(value) -> int:
    if value is None:
        return NULL_SCORE
    try:
        return int(value)
    except (TypeError, ValueError):
        return NULL_SCORE


def _normalize_to_3exam(raw_score: int) -> int:
    if raw_score == NULL_SCORE:
        return NULL_SCORE
    if 310 < raw_score <= 410:
        return round(raw_score * 0.75)
    if raw_score > 410:
        return round(raw_score * 0.60)
    return raw_score


def _can_user_afford(threshold: int, user_score: int) -> bool:
    if threshold == NULL_SCORE:
        return False

    # Случай 1: И у вуза порог за 4 экз./ДВИ (> 310), и юзер ввёл сумму с ДВИ (> 310) —
    # сравниваем их напрямую по 400-балльной шкале (например, 386 <= 390)
    if threshold > 310 and user_score > 310:
        return threshold <= user_score

    # Случай 2: Приводим обе стороны к единой 3-предметной шкале (из 300 баллов),
    # чтобы 320 баллов за 4 экзамена (~240 за 3) не открывали бюджет в МФТИ (273 за 3)
    return _normalize_to_3exam(threshold) <= _normalize_to_3exam(user_score)


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _whole_word_match(needle: str, haystack: str) -> bool:
    needle_words = re.findall(r"[а-яёa-z0-9]+", needle.lower())
    if not needle_words:
        return False
    target = " " + " ".join(needle_words) + " "
    source = " " + " ".join(re.findall(r"[а-яёa-z0-9]+", haystack.lower())) + " "
    return target in source


def _common_prefix_len(w1: str, w2: str) -> int:
    n = 0
    for c1, c2 in zip(w1, w2):
        if c1 != c2:
            break
        n += 1
    return n


def _expand_query(query: str) -> str:
    parts = [query]
    seen = set()
    for _ in range(2):
        current_words = re.findall(r"[а-яёa-z]+", " ".join(parts).lower())
        added = []
        for word in current_words:
            if word in seen:
                continue
            syn = SPECIALTY_SYNONYMS.get(word)
            if not syn:
                continue
            seen.add(word)
            variants = [syn] if isinstance(syn, str) else syn
            for v in variants:
                if v not in parts:
                    added.append(v)
        if not added:
            break
        parts.extend(added)
    if len(parts) == 1:
        return query
    return " ".join(parts)


def _lexical_overlap(query: str, directions: str) -> bool:
    expanded = _expand_query(query)
    q_words = [w for w in re.findall(r"[а-яёa-z]+", expanded.lower())
               if len(w) >= MIN_WORD_LEN]
    d_words = [w for w in re.findall(r"[а-яёa-z]+", directions.lower())
               if len(w) >= MIN_WORD_LEN]

    for qw in q_words:
        for dw in d_words:
            prefix = _common_prefix_len(qw, dw)
            min_len = min(len(qw), len(dw))
            max_len = max(len(qw), len(dw))

            if prefix == max_len:
                return True

            if prefix == min_len:
                if len(qw) < len(dw):
                    return True
                continue

            if prefix >= LONG_PREFIX_LEN and prefix / min_len >= LONG_PREFIX_RATIO:
                return True
            if prefix >= SHORT_PREFIX_MIN and prefix / min_len >= SHORT_PREFIX_RATIO:
                return True
            if (
                len(qw) >= FUZZY_MIN_LEN
                and len(dw) >= FUZZY_MIN_LEN
                and difflib.SequenceMatcher(None, qw, dw).ratio() >= FUZZY_RATIO
            ):
                return True
    return False


def _build_directions_display(
    matched_items: list[dict],
    other_items: list[dict],
) -> tuple[str, str]:
    """Выводит подходящие направления ровной колонкой (без нумерации и сдвигов),
    каждое строго на одной строке и на одном уровне.
    """
    source_items = matched_items if matched_items else other_items[:3]
    if not source_items:
        return "Не указано", "Не указано"

    html_lines: list[str] = []
    plain_parts: list[str] = []

    for item in source_items:
        raw_name = item.get("name", "")
        code = item.get("code", "")
        url = item.get("url", "")

        full_label = f"{code} {raw_name}".strip() if code else raw_name
        plain_parts.append(full_label)

        short_name = _compact_dir_name(raw_name)
        display_label = f"{code} {short_name}".strip() if code else short_name

        if url and url.startswith("http"):
            link_html = f'<a href="{html.escape(url, quote=True)}">{html.escape(display_label)}</a>'
        else:
            link_html = html.escape(display_label)

        html_lines.append(link_html)

    return "\n".join(html_lines), ", ".join(plain_parts)


def _count_matched_directions(
    query: str,
    directions_str: str,
    codes_map: dict[str, str] | None = None,
    dir_items: list[dict] | None = None,
) -> tuple[int, str, str]:
    codes_map = codes_map or {}
    if dir_items is None:
        raw_dirs = [d.strip() for d in directions_str.split(",") if d.strip() and d.strip().lower() != "не указано"]
        dir_items = [{"name": d, "code": "", "url": ""} for d in raw_dirs]

    matched_items: list[dict] = []
    other_items: list[dict] = []

    code_m = _CODE_EXTRACT_RE.search(query)
    if code_m:
        g1, g2, g3 = code_m.groups()
        target_code = f"{g1}.{g2}.{g3}"
        is_group = (g2 == "00" and g3 == "00")
        prefix = f"{g1}."

        for item in dir_items:
            c = item.get("code", "")
            if (is_group and c.startswith(prefix)) or (not is_group and c == target_code):
                matched_items.append(item)
            else:
                other_items.append(item)

        if not matched_items and codes_map:
            matched_names_lc: set[str] = set()
            for c, name in codes_map.items():
                if (is_group and c.startswith(prefix)) or (not is_group and c == target_code):
                    matched_items.append({"name": name, "code": c, "url": ""})
                    matched_names_lc.add(name.lower())
            other_items = [it for it in dir_items if it.get("name", "").lower() not in matched_names_lc]

        matched_items.sort(key=lambda x: _code_level_priority(x.get("code", "")))
        html_str, plain_str = _build_directions_display(matched_items, other_items)
        return len(matched_items), html_str, plain_str

    q_lc = query.lower().strip()
    for item in dir_items:
        d_name = item.get("name", "")
        if _whole_word_match(q_lc, d_name) or _lexical_overlap(q_lc, d_name):
            matched_items.append(item)
        else:
            other_items.append(item)

    matched_items.sort(key=lambda x: _code_level_priority(x.get("code", "")))
    html_str, plain_str = _build_directions_display(matched_items, other_items)
    return len(matched_items), html_str, plain_str


def _get_vuz_tier(univ_name: str, matched_cnt: int = 1, total_dirs_cnt: int = 1) -> int:
    u_lc = " ".join(univ_name.lower().split())
    if "филиал" in u_lc:
        return 0

    if any(p.search(u_lc) for p in _TIER1_PATTERNS):
        if _NARROW_FLAGSHIP_RE.search(u_lc) and matched_cnt == 1 and total_dirs_cnt >= 10:
            return 0
        return 1

    if any(p.search(u_lc) for p in _TIER2_PATTERNS):
        return 2

    return 0


def _effective_budget_score(budget_int: int, paid_int: int, vuz_tier: int) -> int:
    if budget_int == NULL_SCORE:
        return 0

    eff = _normalize_to_3exam(budget_int)
    paid_norm = _normalize_to_3exam(paid_int)

    if vuz_tier == 0 and paid_norm != NULL_SCORE:
        if paid_norm < 145 and (eff - paid_norm) > 85:
            eff = min(eff, paid_norm + 80)

    return eff


def _encode(texts: list[str]) -> np.ndarray:
    vecs = _model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=ENCODE_BATCH_SIZE,
    )
    return np.asarray(vecs, dtype=np.float32)


def _load_index() -> bool:
    global _embeddings, _metadatas
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        return False
    try:
        emb = np.load(EMBEDDINGS_PATH)
        with open(METADATA_PATH, "rb") as f:
            metas = pickle.load(f)
    except Exception:
        logger.exception("Не удалось прочитать файлы индекса")
        _embeddings, _metadatas = None, []
        return False

    if emb.ndim != 2 or emb.shape[1] != EMBEDDING_DIM:
        return False
    if emb.shape[0] != len(metas):
        return False

    _embeddings = emb.astype(np.float32, copy=False)
    _metadatas = metas
    _rebuild_validation_caches()
    return True


def _save_index() -> None:
    if _embeddings is None:
        return
    try:
        np.save(EMBEDDINGS_PATH, _embeddings)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(_metadatas, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        logger.exception("Не удалось сохранить индекс на диск")


def get_index_stats() -> dict:
    idx_time = os.path.getmtime(EMBEDDINGS_PATH) if EMBEDDINGS_PATH.exists() else 0
    return {
        "count": len(_metadatas),
        "last_index_time": idx_time,
    }


def load_data_to_db(json_path: str | Path = JSON_PATH, force: bool = False) -> None:
    global _embeddings, _metadatas
    json_path = Path(json_path)

    if not json_path.exists():
        logger.warning("JSON-файл не найден: %s. Бот запустится с пустой базой.", json_path)
        return

    current_hash = _file_hash(json_path)
    stored_hash = HASH_PATH.read_text().strip() if HASH_PATH.exists() else None
    hash_changed = current_hash != stored_hash

    if not force and not hash_changed:  # noqa: SIM102
        if _load_index() and _embeddings is not None and len(_metadatas) > 0:
            logger.info("Индекс актуален (%d документов)", len(_metadatas))
            return

    logger.info("Перестройка индекса (force=%s, hash_changed=%s)", force, hash_changed)

    t0 = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    metas = []
    for item in data:
        budget = _safe_int(item.get("budget_min_score"))
        paid = _safe_int(item.get("paid_min_score"))
        min_score = (
            NULL_SCORE if budget == NULL_SCORE and paid == NULL_SCORE
            else min(_normalize_to_3exam(budget), _normalize_to_3exam(paid))
        )
        city_raw = canonicalize_city(str(item.get("city") or "Не указан"))
        directions_str = _directions_str(item)
        codes_map = _extract_codes_map(item)
        dir_items = _extract_dir_items(item)

        docs.append(f"passage: {directions_str}")
        metas.append({
            "university": str(item.get("university") or "Без названия"),
            "city": _format_city_title(city_raw),
            "city_norm": normalize_text(city_raw),
            "url": str(item.get("url") or "#"),
            "directions": directions_str,
            "codes_map": codes_map,
            "dir_items_v3": dir_items,
            "budget": str(item.get("budget_min_score")) if item.get("budget_min_score") is not None else "Нет",
            "paid": str(item.get("paid_min_score")) if item.get("paid_min_score") is not None else "Нет",
            "score": min_score,
        })

    embeddings = _encode(docs)

    _embeddings = embeddings
    _metadatas = metas
    _rebuild_validation_caches()
    _save_index()
    HASH_PATH.write_text(current_hash)

    logger.info("Индекс построен: %d документов за %.1f сек.", len(metas), time.time() - t0)


def _format_score_label(val_str: str, prefix: str) -> str:
    if val_str == "Нет":
        return f"{prefix}: нет данных"
    val_int = _safe_int(val_str)
    if val_int != NULL_SCORE and val_int > 310:
        eq = _normalize_to_3exam(val_int)
        return f"{prefix}: от {val_str} <i>(4 экз./ДВИ, ~{eq} за 3)</i>"
    return f"{prefix}: от {val_str}"


def format_one_vuz(m: dict, form: str = "any") -> str:
    univ = html.escape(m["university"])
    url = html.escape(m["url"], quote=True)
    city_name = html.escape(_format_city_title(str(m.get("city") or "")))
    city_badge = f" <i>({city_name})</i>" if city_name else ""

    if m.get("_html_directions"):
        dir_block = f"📚 <b>Подходящие направления:</b>\n{m['_html_directions']}"
    else:
        directions = m.get("_plain_sorted_directions") or m["directions"]
        dir_block = f"📚 {html.escape(directions)}"

    b_str = _format_score_label(str(m.get("budget", "Нет")), "Бюджет")
    p_str = _format_score_label(str(m.get("paid", "Нет")), "Платное")

    if form == "budget":
        b_str = f"<b>{b_str}</b>"
    elif form == "paid":
        p_str = f"<b>{p_str}</b>"

    money = f"💰 {b_str} | {p_str}"

    return f'🏛 <a href="{url}">{univ}</a>{city_badge}\n{dir_block}\n{money}'


def _collect_results_by_form(city: str, specialty: str, score: int) -> dict[str, list]:
    empty = {"any": [], "budget": [], "paid": []}
    if _embeddings is None or len(_metadatas) == 0:
        return empty

    n = len(_metadatas)
    if city.lower() in ("любой", "любой город"):
        mask = np.ones(n, dtype=bool)
    else:
        raw_parts = [p.strip() for p in city.split(",") if p.strip()]
        target_norms = {normalize_text(canonicalize_city(p)) for p in raw_parts}
        mask = np.fromiter(
            (m["city_norm"] in target_norms for m in _metadatas),
            dtype=bool, count=n,
        )

    if not mask.any():
        return empty

    idx = np.flatnonzero(mask)
    is_code_search = bool(_CODE_EXTRACT_RE.search(specialty))

    expanded_specialty = _expand_query(specialty)
    query_vec = _encode([f"query: {expanded_specialty}"])[0]
    sub = _embeddings[idx]
    sims = sub @ query_vec
    dists = 1.0 - sims

    specialty_lc = specialty.lower().strip()
    list_any = []
    list_budget = []
    list_paid = []

    for j, i in enumerate(idx):
        meta = dict(_metadatas[int(i)])
        budget_int = _safe_int(meta.get("budget"))
        paid_int = _safe_int(meta.get("paid"))

        can_afford_budget = _can_user_afford(budget_int, score)
        can_afford_paid = _can_user_afford(paid_int, score)
        has_any_budget = (budget_int != NULL_SCORE)

        if not (can_afford_budget or can_afford_paid):
            continue

        directions = meta["directions"]
        if not directions or directions.lower() == "не указано":
            continue

        codes_map = meta.get("codes_map") or {}
        dir_items = meta.get("dir_items_v3")
        distance = float(dists[j])

        matched_cnt, html_dirs, plain_dirs = _count_matched_directions(
            specialty_lc, directions, codes_map, dir_items
        )
        total_dirs_cnt = len(dir_items) if dir_items is not None else len(directions.split(","))

        if is_code_search:
            if matched_cnt == 0:
                continue
            is_text_match = True
            has_overlap = True
            is_semantic_match = True
        else:
            is_text_match = _whole_word_match(specialty_lc, directions)
            has_overlap = matched_cnt > 0
            is_semantic_match = distance < SEMANTIC_DISTANCE_LIMIT and has_overlap
            if not (is_text_match or has_overlap or is_semantic_match):
                continue

        meta["_html_directions"] = html_dirs
        meta["_plain_sorted_directions"] = plain_dirs

        if can_afford_budget:
            budget_tier = 0
        elif has_any_budget:
            budget_tier = 1
        else:
            budget_tier = 2

        univ_lc = meta["university"].lower()
        is_branch = 1 if ("филиал" in univ_lc and "филиал" not in specialty_lc) else 0
        match_tier = 0 if (is_text_match or matched_cnt > 0) else 1

        vuz_tier = _get_vuz_tier(meta["university"], matched_cnt, total_dirs_cnt)
        tier_order = 0 if vuz_tier == 1 else (1 if vuz_tier == 2 else 2)

        eff_budget = _effective_budget_score(budget_int, paid_int, vuz_tier)
        paid_norm = _normalize_to_3exam(paid_int) if paid_int != NULL_SCORE else 0

        profile_bonus = min(matched_cnt, 5) * 3
        paid_tiebreaker = paid_norm // 25
        non_profile_penalty = (
            -35
            if (not is_code_search and matched_cnt == 1 and total_dirs_cnt >= 8 and vuz_tier == 0)
            else 0
        )

        base_prestige_any = eff_budget if can_afford_budget else paid_norm
        rank_power_any = base_prestige_any + profile_bonus + paid_tiebreaker + non_profile_penalty

        meta["_sort_key_any"] = (
            match_tier,
            is_branch,
            budget_tier,
            tier_order,
            -rank_power_any,
            round(distance, 1),
        )

        base_prestige_paid = max(eff_budget, paid_norm)
        rank_power_paid = base_prestige_paid + profile_bonus + paid_tiebreaker + non_profile_penalty
        paid_state_tier = 0 if has_any_budget else 1

        meta["_sort_key_paid"] = (
            match_tier,
            is_branch,
            paid_state_tier,
            tier_order,
            -rank_power_paid,
            round(distance, 1),
        )

        list_any.append(meta)
        if can_afford_budget:
            list_budget.append(meta)
        if can_afford_paid:
            list_paid.append(meta)

    list_any.sort(key=lambda m: m["_sort_key_any"])
    list_budget.sort(key=lambda m: m["_sort_key_any"])
    list_paid.sort(key=lambda m: m["_sort_key_paid"])

    return {
        "any": list_any,
        "budget": list_budget,
        "paid": list_paid,
    }


def search_vuz_by_forms(city: str, specialty: str, score: int) -> dict[str, list]:
    try:
        return _collect_results_by_form(city, specialty, score)
    except Exception:
        logger.exception("Поиск упал - попробую перестроить индекс")
        try:
            HASH_PATH.unlink(missing_ok=True)
            load_data_to_db(force=True)
            return _collect_results_by_form(city, specialty, score)
        except Exception:
            return {"any": [], "budget": [], "paid": []}


def search_vuz(city: str, specialty: str, score: int, form: str = "any") -> list:
    res = search_vuz_by_forms(city, specialty, score)
    return res.get(form, res["any"])