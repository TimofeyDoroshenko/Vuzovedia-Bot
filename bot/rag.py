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

# Порог семантической близости (distance = 1 - косинус).
# Семантическая близость принимается только если дополнительно есть
# лексический якорь (стем-совпадение). Расширение через словарь синонимов
# само по себе НЕ является якорем.
SEMANTIC_DISTANCE_LIMIT = 0.35

# Минимальная длина слова для учёта в лексическом совпадении.
MIN_WORD_LEN = 5

# Правило лексического совпадения двух слов:
#   - длинное правило: префикс >= LONG_PREFIX_LEN и его доля от длины
#     короткого слова >= LONG_PREFIX_RATIO;
#   - короткое правило: префикс >= SHORT_PREFIX_MIN и его доля от длины
#     короткого слова >= SHORT_PREFIX_RATIO;
#   - fuzzy для 8+ букв: SequenceMatcher ratio >= FUZZY_RATIO.
LONG_PREFIX_LEN = 8
LONG_PREFIX_RATIO = 0.8
SHORT_PREFIX_MIN = 4
SHORT_PREFIX_RATIO = 0.9

FUZZY_MIN_LEN = 8
FUZZY_RATIO = 0.87

logger.info("Загружаю embedding-модель...")
_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
logger.info("Embedding-модель загружена.")

_embeddings = None
_metadatas = []

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

# Словарь синонимов: пользовательские слова -> корни, которые реально
# встречаются в названиях направлений. Значение может быть строкой или
# списком строк - если слово маппится сразу на несколько областей.
# Используется при раскрытии запроса перед лексической проверкой.
SPECIALTY_SYNONYMS = {
    # ─── Русские синонимы ───
    "лечить": "лечебное",
    "лечение": "лечебное",
    "лечу": "лечебное",
    "врач": "лечебное",
    "медик": "лечебное",
    "медицина": "лечебное",
    "медицинский": "лечебное",
    "айти": ["информатика", "программная"],
    "ит": ["информатика", "программная"],
    # Программирование относится и к Программной инженерии, и к Информатике.
    "программирование": ["программная", "информатика"],
    "программировать": ["программная", "информатика"],
    "программист": ["программная", "информатика"],
    "разработчик": ["программная", "информатика"],
    "кодер": ["программная", "информатика"],
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
    # Медицина: падежи и производные
    "хирургия": "лечебное",
    "хирург": "лечебное",
    "хирургический": "лечебное",
    "терапия": "лечебное",
    "терапевт": "лечебное",
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
    # Опечатки, которые мы хотим ловить через словарь
    "програмирование": ["программная", "информатика"],
    # ─── IT-термины на латинице ───
    "it": ["информатика", "программная"],
    "qa": ["программная", "информатика"],
    "ai": "информатика",
    "ml": "информатика",
    "web": ["программная", "информатика"],
    "data": "информатика",
    "sql": "информатика",
    "python": ["программная", "информатика"],
    "java": ["программная", "информатика"],
    "javascript": ["программная", "информатика"],
    "kotlin": ["программная", "информатика"],
    "swift": ["программная", "информатика"],
    "golang": ["программная", "информатика"],
    "frontend": ["программная", "информатика"],
    "backend": ["программная", "информатика"],
    "fullstack": ["программная", "информатика"],
    "devops": "информатика",
    "ux": "дизайн",
    "ui": "дизайн",
    "c++": ["программная", "информатика"],
    "c#": ["программная", "информатика"],
    "data science": "информатика",
    "datascience": "информатика",
    "датасаенс": "информатика",
}


def normalize_text(text: str) -> str:
    return text.lower().replace("-", "").replace(" ", "").strip()


def canonicalize_city(name: str) -> str:
    key = normalize_text(name)
    return CITY_ALIASES.get(key, name.strip())


def get_proper_city_name(city: str) -> str:
    if city.lower() == "любой":
        return "Любой город"
    base_norm = normalize_text(canonicalize_city(city))
    if not _metadatas:
        return city.strip().title()
    norm_to_real = {m["city_norm"]: m["city"] for m in _metadatas}
    if base_norm in norm_to_real:
        return norm_to_real[base_norm]
    matches = difflib.get_close_matches(base_norm, norm_to_real.keys(), n=1, cutoff=0.8)
    if matches:
        return norm_to_real[matches[0]]
    return city.strip().title()


def _safe_int(value) -> int:
    if value is None:
        return NULL_SCORE
    try:
        return int(value)
    except (TypeError, ValueError):
        return NULL_SCORE


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _directions_str(item: dict) -> str:
    dl = item.get("directions_list") or []
    names, seen = [], set()
    for d in dl:
        name = (d.get("name") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    if names:
        return ", ".join(names)
    return str(item.get("directions") or "Не указано")


def _whole_word_match(needle: str, haystack: str) -> bool:
    needle_words = re.findall(r"[а-яёa-z0-9]+", needle.lower())
    if not needle_words:
        return False
    target = " " + " ".join(needle_words) + " "
    source = " " + " ".join(re.findall(r"[а-яёa-z0-9]+", haystack.lower())) + " "
    return target in source


def _common_prefix_len(w1: str, w2: str) -> int:
    """Длина общего префикса двух строк."""
    n = 0
    for c1, c2 in zip(w1, w2):
        if c1 != c2:
            break
        n += 1
    return n


def _expand_query(query: str) -> str:
    """Раскрывает синонимы в запросе (в т.ч. транзитивно, до 2 уровней).

    Значение в SPECIALTY_SYNONYMS может быть строкой или списком строк -
    это позволяет маппить одно слово на несколько корней сразу
    ("программирование" -> ["программная", "информатика"]).
    """
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
    """Лексическое совпадение слов запроса и слов направлений.

    Слова короче MIN_WORD_LEN игнорируются. Логика проверки пары слов:

      1) слова равны;
      2) префикс покрывает всё короткое слово, И это слово - запрос
         (пользователь написал корень, направление его содержит:
         "юрист" / "юриспруденция"). Обратный случай, когда короткое
         слово - это направление, а запрос длиннее и начинается с
         него, отклоняем: это мусор вида "программбурмалда" / "программ";
      3) обычное префиксное совпадение:
         - длинное правило: префикс >= LONG_PREFIX_LEN и ratio >= LONG_PREFIX_RATIO;
         - короткое правило: префикс >= SHORT_PREFIX_MIN и ratio >= SHORT_PREFIX_RATIO;
         - fuzzy для 8+ букв: SequenceMatcher ratio >= FUZZY_RATIO.
    """
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

            # 1) Полное совпадение слов.
            if prefix == max_len:
                return True

            # 2) Префикс покрывает всё короткое слово.
            if prefix == min_len:
                # OK, только если короткое слово - это запрос.
                if len(qw) < len(dw):
                    return True
                # Иначе это мусор. Пропускаем пару.
                continue

            # 3) Обычное префиксное совпадение.
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
            else min(budget, paid)
        )
        city_raw = canonicalize_city(str(item.get("city") or "Не указан"))
        directions_str = _directions_str(item)

        docs.append(f"passage: {directions_str}")
        metas.append({
            "university": str(item.get("university") or "Без названия"),
            "city": city_raw.title(),
            "city_norm": normalize_text(city_raw),
            "url": str(item.get("url") or "#"),
            "directions": directions_str,
            "budget": str(item.get("budget_min_score")) if item.get("budget_min_score") is not None else "Нет",
            "paid": str(item.get("paid_min_score")) if item.get("paid_min_score") is not None else "Нет",
            "score": min_score,
        })

    embeddings = _encode(docs)

    _embeddings = embeddings
    _metadatas = metas
    _save_index()
    HASH_PATH.write_text(current_hash)

    logger.info("Индекс построен: %d документов за %.1f сек.", len(metas), time.time() - t0)


def format_one_vuz(m: dict) -> str:
    univ = html.escape(m["university"])
    url = html.escape(m["url"], quote=True)

    directions = m["directions"]
    if len(directions) > 350:
        directions = directions[:350].rsplit(",", 1)[0] + " и др."
    dir_e = html.escape(directions)

    budget = m["budget"]
    paid = m["paid"]
    money = "💰 "
    money += "Бюджет: от " + budget if budget != "Нет" else "Бюджет: нет данных"
    money += " | "
    money += "Платное: от " + paid if paid != "Нет" else "Платное: нет данных"

    return f'🏛 <a href="{url}">{univ}</a>\n📚 {dir_e}\n{money}'


def _collect_results(city: str, specialty: str, score: int) -> list:
    if _embeddings is None or len(_metadatas) == 0:
        return None

    n = len(_metadatas)
    scores = np.fromiter((m["score"] for m in _metadatas), dtype=np.int32, count=n)
    if city.lower() == "любой":
        mask = scores <= score
    else:
        proper_city = get_proper_city_name(city)
        city_norm = normalize_text(proper_city)
        city_ok = np.fromiter(
            (m["city_norm"] == city_norm for m in _metadatas),
            dtype=bool, count=n,
        )
        mask = city_ok & (scores <= score)

    if not mask.any():
        return None

    idx = np.flatnonzero(mask)
    expanded_specialty = _expand_query(specialty)
    query_vec = _encode([f"query: {expanded_specialty}"])[0]
    sub = _embeddings[idx]
    sims = sub @ query_vec
    dists = 1.0 - sims

    specialty_lc = specialty.lower().strip()
    valid = []

    for j, i in enumerate(idx):
        meta = dict(_metadatas[int(i)])
        directions = meta["directions"]
        distance = float(dists[j])

        is_text_match = _whole_word_match(specialty_lc, directions)
        has_overlap = _lexical_overlap(specialty_lc, directions)
        is_semantic_match = distance < SEMANTIC_DISTANCE_LIMIT and has_overlap

        if is_text_match or is_semantic_match:
            meta["_exact"] = 0 if is_text_match else 1
            meta["_distance"] = distance
            valid.append(meta)

    if not valid:
        return None

    valid.sort(key=lambda m: (m["_exact"], m["_distance"], -m["score"]))
    return valid


def search_vuz(city: str, specialty: str, score: int) -> list:
    try:
        metas = _collect_results(city, specialty, score)
    except Exception:
        logger.exception("Поиск упал - попробую перестроить индекс")
        try:
            HASH_PATH.unlink(missing_ok=True)
            load_data_to_db(force=True)
            metas = _collect_results(city, specialty, score)
        except Exception:
            return []
    if not metas:
        return []
    return metas