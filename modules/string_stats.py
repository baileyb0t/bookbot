# ---- dependencies {{{
import re
from typing import Dict, List

# }}}


# --- support methods --- {{{
def get_num_words(book: str) -> int:
    if book in (None, ""):
        return 0
    assert type(book) is str
    words = book.split()
    return len(words)


def get_num_chars(book: str) -> Dict:
    bookstr = book.lower()
    chars = set(book)
    charcount = {
        char: len(re.findall(pattern=re.escape(char), string=bookstr)) for char in chars
    }
    return charcount


def get_count(item: Dict):
    return item["num"]


def get_sorted_charcount(cc: Dict) -> List[Dict]:
    neat = [
        {"char": char, "num": count}
        for char, count in cc.items()
        if (char != " ") & (count != 0)
    ]
    neat.sort(reverse=True, key=get_count)
    return neat


# }}}
