import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import rag

rag.load_data_to_db()

print("=== КОНФИГИ ===")
print("MIN_WORD_LEN:", getattr(rag, "MIN_WORD_LEN", "НЕТ"))
print("LONG_PREFIX_LEN:", getattr(rag, "LONG_PREFIX_LEN", "НЕТ"))
print("SHORT_PREFIX_MIN:", getattr(rag, "SHORT_PREFIX_MIN", "НЕТ"))
print("SHORT_PREFIX_RATIO:", getattr(rag, "SHORT_PREFIX_RATIO", "НЕТ"))
print("Есть _prefix_ratio (старое):", hasattr(rag, "_prefix_ratio"))
print("Есть _common_prefix_len (новое):", hasattr(rag, "_common_prefix_len"))
print()

q = "програмбурмалда"
expanded = rag._expand_query(q)
print(f"q = {q!r}")
print(f"expanded = {expanded!r}")
print()

for meta in rag._metadatas:
    if rag._lexical_overlap(q, meta["directions"]):
        print(f"ПЕРВЫЙ МАТЧ: {meta['university']}")
        print(f"directions: {meta['directions'][:250]}")
        print()

        q_words = [w for w in re.findall(r"[а-яёa-z]+", expanded.lower())
                   if len(w) >= rag.MIN_WORD_LEN]
        d_words = [w for w in re.findall(r"[а-яёa-z]+", meta["directions"].lower())
                   if len(w) >= rag.MIN_WORD_LEN]

        print(f"q_words ({len(q_words)}): {q_words}")
        print(f"d_words (первые 40): {d_words[:40]}")
        print()

        if hasattr(rag, "_common_prefix_len"):
            for qw in q_words:
                for dw in d_words:
                    p = rag._common_prefix_len(qw, dw)
                    if p >= 3:
                        min_len = min(len(qw), len(dw))
                        print(f"  q={qw!r:25} d={dw!r:25} prefix={p} ratio={p/min_len:.2f}")
        break