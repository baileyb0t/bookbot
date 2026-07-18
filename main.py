#!/usr/bin/env python3
# vim: set ts=4 sts=0 sw=4 si fenc=utf-8 et:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:
# Authors:     BP
# Maintainers: BP
# =========================================

# ---- dependencies {{{
import argparse
import os
import subprocess
import sys
from pathlib import Path
from random import randint

import pandas as pd
from loguru import logger
from mdutils import MdUtils

sys.path.append("modules")
from llm import hashstr, langrequest, msgs_from_text, setupagent, writeprompt
from string_stats import get_num_words

# }}}


# --- support methods --- {{{
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/books/frankenstein.txt")
    parser.add_argument(
        "--delimiter", default="CHAPTER"
    )  # Try "PAGE# 1" for DPA Reports
    parser.add_argument("--datatype", default="Event")
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()
    assert Path(args.input).exists()
    assert Path(args.outdir).exists()
    return args


def setuplogging(logfile):
    logger.add(
        logfile,
        colorize=True,
        format="<green>{time:YYYY-MM-DD⋅at⋅HH:mm:ss}</green>⋅<level>{message}</level>",
        level="INFO",
    )
    return 1


def getfiles(arg, fext) -> list:
    assert os.path.isdir(arg)
    return [path for path in Path(arg).rglob(f"*.{fext}")]


def readtext(filepath):
    assert Path(filepath).exists, f"{filepath} could not be found"
    with open(filepath, "r") as f:
        contents = "".join(f.readlines())
    return contents


def resolveinput(filepath):
    if os.path.isfile(filepath):
        return filepath
    cands = getfiles(arg=filepath, fext="txt")
    return cands[randint(0, len(cands) - 1)]


def gettitle(file):
    name = Path(file).stem
    name = name.replace("-", " ").replace("_", " ").title()
    return name


def consolidate_delimiter(text, delim, opts):
    for opt in opts:
        text = text.replace(opt, delim)
    return text


def process(agent, prompt: str, info: dict) -> pd.DataFrame:
    msgs = msgs_from_text(prompt=prompt, text=info["text"], sep=info["delimiter"])
    nmessages = len(msgs)
    responses = []
    for i in range(nmessages):
        logger.info(f"processing message {i + 1} out of {nmessages}")
        msg = msgs[i]
        resp = langrequest(agent=agent, messages={"messages": msg})
        if resp is not None:
            resp_data = pd.DataFrame([resp["structured_response"].model_dump()])
        else:
            resp_data = pd.DataFrame([{"error_no_response": True}])
        resp_data["textid"] = info["textid"]
        resp_data["filename"] = info["textpath"]
        resp_data["prompt"] = prompt
        resp_data["section"] = msg["content"].split(":** ")[1]
        resp_data["sectionid"] = resp_data.section.apply(hashstr)
        resp_data["recordid"] = (
            resp_data.textid.str[:6] + "_" + resp_data.sectionid.str[:6]
        )
        responses.append(resp_data)
    out = pd.concat(responses)
    out.reset_index(drop=True, inplace=True)
    return out


def makereport_md(info):
    """Should we be using new_table instead of new_paragraph w/ html passed?
    Guess is no because we don't want this script to bother too much
    with what appears in the html - since those columns are based on the requested
    data type extracted and that's handled by a separate module.
    BUT if the rendered html paragraph doesn't look right,
    shouldn't be too hard to write a wrapper to handle reformatting
    for the sample record and first 5 rows.
    """
    reporttitle = f"LLM-aided report on {info['title']}"
    report = MdUtils(file_name=f"{outdir}/book_report", title=reporttitle)

    report.new_header(level=1, title="Report")
    report.new_line(
        f"Analyzing text {info['textid']} found at {info['textpath']}...",
        bold_italics_code="cib",
        align="center",
        color="gray",
    )

    report.new_header(level=2, title="Basics")
    report.new_paragraph(f"""\n
    - {info["wordcount"]} total words
    - Estimated {info["token_est"]:,} tokens
    - {len(info["text"].split("\x0c")):,} pages (using unicode page delimiter)
    """)

    report.new_header(level=2, title="Extractions via LLM")
    report.new_header(level=3, title="General info")
    report.new_paragraph(f"""
    - Text divided into {info["responses"].shape[0]:,} sections using delimiter "{info["delimiter"]}"
    - 1 section per message to LLM
    """)
    report.new_header(level=3, title=f"Sample extracted {args.datatype} record")
    report.new_paragraph(info["responses"].sample().T.to_html())
    report.new_paragraph()
    report.new_header(level=3, title=f"First 5 {args.datatype}s extracted")
    report.new_paragraph(info["responses"].head().to_html())

    report.new_paragraph("---")
    report.new_header(level=1, title="Text analyzed (first 5,000 characters)")
    report.new_paragraph(info["text"][:5000])
    report.new_paragraph("---")

    report.new_table_of_contents(table_title="Contents", depth=2)
    return report


# }}}

# --- main --- {{{
args = getargs()
setuplogging("logs/main.log")

infile = resolveinput(filepath=args.input)
text = readtext(filepath=infile)[:5000]
text = consolidate_delimiter(
    text, delim=args.delimiter, opts=["PAGE #1", "Page# 1", "Page #1"]
)

logger.info("establishing basic information")
info = {
    "textid": hashstr(text),
    "textpath": args.input,
    "text": text,
    "delimiter": args.delimiter,
    "token_est": len(text),
    "wordcount": get_num_words(text),
    "title": gettitle(file=args.input),
}
logger.info(f"setting up model and agent for extracting {args.datatype} data")
prompt = writeprompt(datatype=args.datatype, sep=":** ")
logger.info(f"applying prompt\n{prompt}")
info["prompt"] = prompt
agent = setupagent(prompt=prompt, datatype=args.datatype)
responses = process(agent=agent, prompt=prompt, info=info)
sections = responses[["textid", "sectionid", "section"]]
responses.drop(columns="section", inplace=True)
info["responses"] = responses

# @TODO: handle existing outputs more intentionally
outdir = f"{args.outdir}/{info['textid']}"
if not Path(outdir).exists():
    subprocess.call(["mkdir", outdir])
else:
    print(
        f"\n\nWARNING: Textid {
            info['textid']
        } found in output directory already\n\nContents:\n"
    )
    print(os.listdir(outdir))

report = makereport_md(info)
report.create_md_file()

sections.to_parquet(f"{outdir}/{args.datatype}_sections.parquet")
responses.to_parquet(f"{outdir}/{args.datatype}_responses.parquet")

logger.info("done")
# }}}
