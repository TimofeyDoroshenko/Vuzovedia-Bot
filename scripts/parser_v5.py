# fix_fast.py
"""
Параллельно догружает с личных страниц Вузопедии:
  1) ПОЛНЫЕ названия вузов (из <h1> или og:title / <title> для старой вёрстки);
  2) ВСЕ специальности (из модалок edOverviewBases-vo-{id}-*).

Прогоняет всё через ThreadPoolExecutor(3), с бэкапом и автосейвом.
"""
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from bs4 import BeautifulSoup
from curl_cffi import requests

# ─── КОНФИГ ────────────────────────────────────────────────────────────────
BASE_URL    = "https://vuzopedia.ru"
INPUT_PATH  = Path("vuz_data_full.json")           # ← поправь, если имя другое
OUTPUT_PATH = Path("vuz_data_full.json")
BACKUP_PATH = Path("vuz_data_full.backup.json")

WORKERS         = 3        # параллельных воркеров
MAX_RETRIES     = 3
SAVE_EVERY      = 50       # автосейв каждые N завершённых вузов
JITTER_MIN      = 0.2      # пауза между задачами в главном потоке
JITTER_MAX      = 0.6
# ───────────────────────────────────────────────────────────────────────────

_print_lock = Lock()
_save_lock = Lock()


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


# ─── Извлечение полного названия ───────────────────────────────────────────
def extract_full_name(soup: BeautifulSoup) -> str | None:
    """
    Пытается достать полное название вуза в порядке приоритета:
      1) <h1 class="mainTitle fc-white">  (современная вёрстка)
      2) любой <h1>                       (если классы отличаются)
      3) og:title                         (универсальный fallback)
      4) <title>                          (последний шанс)
    Отсеивает обрезки, оканчивающиеся на '..'.
    """
    # 1–2) h1-варианты
    for sel in ("h1.mainTitle.fc-white", "h1.mainTitle", "h1"):
        h1 = soup.select_one(sel)
        if not h1:
            continue
        name = _clean(h1.get_text())
        if name and len(name) > 5 and not name.endswith(".."):
            return name

    # 3) og:title
    og = soup.select_one('meta[property="og:title"]')
    if og:
        content = _clean(og.get("content", ""))
        # "Имя (Короткое) 2026: стоит ли поступать?"
        m = re.match(r"^(.+?)\s*\([^)]+\)\s*\d{4}", content)
        if m:
            return m.group(1).strip()
        # "Имя 2026: ..."
        m = re.match(r"^(.+?)\s*\d{4}\s*[:—\-]", content)
        if m:
            return m.group(1).strip()

    # 4) <title>
    t = soup.select_one("title")
    if t:
        raw = _clean(t.get_text())
        m = re.match(r"^[^:]+:\s*(.+?)\s*[—\-]\s*стоит", raw)
        if m:
            return m.group(1).strip()

    return None


# ─── Извлечение специальностей ─────────────────────────────────────────────
def extract_specialties(soup: BeautifulSoup, vuz_id: str) -> list[dict]:
    """
    Тянет ВСЕ специальности из модалок edOverviewBases-vo-{vuz_id}-*.
    Каждая запись: name, code, group, level, url.
    """
    result, seen = [], set()

    for modal in soup.select(f'[id^="edOverviewBases-vo-{vuz_id}-"]'):
        title_el = modal.select_one(".modal-title")
        level = _clean(title_el.get_text()) if title_el else ""

        for section in modal.select(".ed-overview-modal-section"):
            group_el = section.select_one(".ed-overview-modal-section__title")
            group = _clean(group_el.get_text()) if group_el else ""

            for a in section.select(".ed-overview-modal__list a"):
                href = a.get("href", "").strip()
                text = _clean(a.get_text())
                if not text or not href:
                    continue

                # "Название (01.03.02)"
                m = re.match(r"^(.*?)\s*\((\d{2}\.\d{2}\.\d{2})\)\s*$", text)
                if m:
                    name, code = m.group(1).strip(), m.group(2)
                else:
                    name, code = text, ""

                key = (code, name.lower())
                if key in seen:
                    continue
                seen.add(key)

                result.append({
                    "name": name,
                    "code": code,
                    "group": group,
                    "level": level,
                    "url": href if href.startswith("http") else BASE_URL + href,
                })

    return result


# ─── Обработка одного вуза ─────────────────────────────────────────────────
def fetch_one(vuz: dict, idx: int, total: int) -> dict:
    url = vuz.get("url")
    if not url:
        return {"__skip__": True}

    vuz_id = url.rstrip("/").split("/")[-1]
    label = (vuz.get("university") or vuz_id)[:45]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            s = requests.Session(impersonate="chrome", timeout=30)
            r = s.get(url, timeout=30)

            if r.status_code == 429:
                wait = 5 * attempt
                with _print_lock:
                    print(f"   ⏸ 429, ждём {wait}s ({label})")
                time.sleep(wait)
                continue

            if r.status_code != 200:
                with _print_lock:
                    print(f"   ⚠️ HTTP {r.status_code} ({label})")
                time.sleep(2)
                continue

            soup = BeautifulSoup(r.text, "lxml")

            full_name = extract_full_name(soup)
            specs = extract_specialties(soup, vuz_id)

            with _print_lock:
                name_info = f" ✔ {full_name[:60]}" if full_name else " ❌ нет имени"
                spec_info = f" 📚 {len(specs)}" if specs else " ⚠️ 0 спец."
                print(f"[{idx}/{total}] {label}{name_info}{spec_info}")

            return {
                "university": full_name or vuz.get("university"),
                "directions_list": specs,
                "directions": ", ".join(s["name"] for s in specs)
                              if specs else vuz.get("directions", "Не указано"),
            }

        except Exception as e:
            with _print_lock:
                print(f"   ❌ попытка {attempt}: {e} ({label})")
            time.sleep(2 * attempt)

    with _print_lock:
        print(f"   ❌ сдался после {MAX_RETRIES} попыток ({label})")
    return {"__failed__": True}


def save_progress(data: list[dict]):
    with _save_lock:
        OUTPUT_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


# ─── main ──────────────────────────────────────────────────────────────────
def main():
    if not INPUT_PATH.exists():
        print(f"❌ Не нашёл {INPUT_PATH}")
        return

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    # Бэкап
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"💾 Бэкап → {BACKUP_PATH}")
    else:
        print(f"ℹ️  Бэкап уже есть, не перезаписываю: {BACKUP_PATH}")

    total = len(data)
    est_min = max(1, total * 2 // WORKERS // 60)
    print(f"🚀 {total} вузов, {WORKERS} воркеров, оценка ~{est_min}–{est_min * 2} мин\n")

    fixed_titles = 0
    failed = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(fetch_one, vuz, i, total): i - 1
            for i, vuz in enumerate(data, 1)
        }

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                patch = fut.result()
            except Exception as e:
                print(f"   ❌ необработанная ошибка: {e}")
                failed += 1
                completed += 1
                continue

            if patch.get("__skip__"):
                completed += 1
                continue

            if patch.get("__failed__"):
                failed += 1
                completed += 1
                continue

            old_name = data[idx].get("university")
            if patch.get("university") and patch["university"] != old_name:
                data[idx]["university"] = patch["university"]
                fixed_titles += 1

            if patch.get("directions_list"):
                data[idx]["directions_list"] = patch["directions_list"]
                data[idx]["directions"] = patch["directions"]

            completed += 1
            if completed % SAVE_EVERY == 0:
                save_progress(data)
                with _print_lock:
                    print(f"   💾 сохранено ({completed}/{total})")

            time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))

    save_progress(data)

    print(f"\n✅ Готово.")
    print(f"   Названий обновлено: {fixed_titles}")
    print(f"   Ошибок            : {failed}")
    print(f"   Файл              : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()