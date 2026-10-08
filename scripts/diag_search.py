import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import rag

rag.load_data_to_db()

POSITIVES = [
    ("лечить людей", "Лечебное дело"),
    ("программирование", "Программная инженерия"),
    ("айти", "Информатика и вычислительная техника"),
    ("рисование", "Дизайн"),
    ("строить дома", "Строительство"),
    ("экономика", "Экономика"),
    ("юрист", "Юриспруденция"),
    ("физика", "Физика"),
    ("врач", "Лечебное дело"),
    ("учитель", "Педагогическое образование"),
    ("it", "Информатика и вычислительная техника"),
    ("ml", "Информатика и вычислительная техника"),
    ("web", "Программная инженерия"),
    ("ui", "Дизайн"),
    ("ux", "Дизайн"),
    ("python", "Программная инженерия"),
    ("sql", "Информатика и вычислительная техника"),
]

NEGATIVES = [
    "фывафыва",
    "абракадабра",
    "купить слона",
    "рецепт борща",
    "как дела",
    "погода на завтра",
    "стихи про осень",
    "привет как ты",
    "хочу спать",
    "где тут кофе",
    "програмбурмалда",
    "программбурмалда",
    "переустановка винды",
    "случайный набор букв",
]


def find_target_meta(target):
    for m in rag._metadatas:
        if target.lower() in m["directions"].lower():
            return m
    return None


def main():
    print("=" * 78)
    print("КАЛИБРОВКА (длинное правило: prefix >= 8 И ratio >= 0.8)")
    print("=" * 78)

    print("\n--- ПОЗИТИВЫ ---\n")
    pos_ok = 0
    for query, target in POSITIVES:
        found = find_target_meta(target)
        if not found:
            print(f"[X] {query!r} -> {target!r}: цель не найдена в базе")
            continue

        idx = rag._metadatas.index(found)
        expanded = rag._expand_query(query)
        q_vec = rag._encode([f"query: {expanded}"])[0]
        dist = 1 - float(rag._embeddings[idx] @ q_vec)
        has_overlap = rag._lexical_overlap(query, found["directions"])
        is_text_match = rag._whole_word_match(query.lower(), found["directions"])

        ok = is_text_match or (dist < rag.SEMANTIC_DISTANCE_LIMIT and has_overlap)
        pos_ok += int(ok)
        marker = "[OK]" if ok else "[FAIL]"
        print(f"{marker} {query!r:20} -> {target!r:35}")
        print(f"      distance={dist:.3f}  overlap={has_overlap}")

    print(f"\nПозитивы: {pos_ok}/{len(POSITIVES)}\n")

    print("--- НЕГАТИВЫ ---\n")
    neg_ok = 0
    for query in NEGATIVES:
        expanded = rag._expand_query(query)
        q_vec = rag._encode([f"query: {expanded}"])[0]
        dists = 1 - rag._embeddings @ q_vec

        passed = sum(
            1 for i, d in enumerate(dists)
            if d < rag.SEMANTIC_DISTANCE_LIMIT
            and rag._lexical_overlap(query, rag._metadatas[i]["directions"])
        )
        if passed == 0:
            neg_ok += 1
            print(f"[OK] {query!r:30} чисто")
        else:
            print(f"[!] {query!r:30} пройдёт как {passed} вузов")

    print(f"\nНегативы чистые: {neg_ok}/{len(NEGATIVES)}")

    print("\n" + "=" * 78)
    if pos_ok == len(POSITIVES) and neg_ok == len(NEGATIVES):
        print("Идеально.")
    else:
        print(f"Позитивы: {pos_ok}/{len(POSITIVES)}, "
              f"негативы: {neg_ok}/{len(NEGATIVES)}")


if __name__ == "__main__":
    main()