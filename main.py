#!/usr/bin/env python3
# vim: set ts=4 sts=0 sw=4 si fenc=utf-8 et:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:
# Authors:     BP
# Maintainers: BP
# =========================================

# ---- dependencies {{{
import argparse
from pathlib import Path

from stats import get_num_chars, get_num_words, get_sorted_charcount

# }}}


# --- support methods --- {{{
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    assert Path(args.input).exists()
    return args


def get_book_text(filepath):
    assert Path(filepath).exists, f"{filepath} could not be found"
    with open(filepath, "r") as f:
        contents = "".join(f.readlines())
    return contents


# }}}

# --- main --- {{{
args = getargs()
bookpath = args.input
book = get_book_text(filepath=bookpath)
wc = get_num_words(book=book)
cc = get_num_chars(book=book)
sortedcc = get_sorted_charcount(cc=cc)

report = f"""\
============ BOOKBOT ============
Analyzing book found at {bookpath}...
----------- Word Count ----------
Found {wc} total words
--------- Character Count -------
"""
for charinfo in sortedcc:
    report += f"{charinfo['char']}: {charinfo['num']}\n"
report += "============= END ==============="
print(report)
# }}}
