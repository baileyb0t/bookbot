#!/usr/bin/env python3
# vim: set ts=4 sts=0 sw=4 si fenc=utf-8 et:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:
# Authors:     BP
# Maintainers: BP
# =========================================

# ---- dependencies {{{
import argparse
import hashlib
import os
import sys
from os.path import isdir, isfile
from pathlib import Path

import langchain_mistralai
import pandas as pd
from langchain.agents import create_agent
from loguru import logger

sys.path.append("../modules")
from fields import AllegationInfo, Contact, Event, Mention

MODEL_ID = "ministral-3b-2512"
MAX_CHUNK_SIZE = 2048
PROMPT_SEPARATOR = ":** "
TEXT_DELIMITER = "PAGE# 1"
OUTPUT_DIR = "output"
OUTPUT_TYPES = {
    "CONTACT": Contact,
    "MENTION": Mention,
    "EVENT": Event,
    "ALLEGATION": AllegationInfo,
}
# }}}


# --- support methods --- {{{
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--delimiter", default=TEXT_DELIMITER)
    parser.add_argument("--datatype", default=None)
    parser.add_argument("--output", default=f"{OUTPUT_DIR}/responses.parquet")
    args = parser.parse_args()
    assert Path(args.input).exists()
    assert args.datatype.upper() in OUTPUT_TYPES.keys()
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


def readtext(fname) -> str:
    with open(fname, "rt") as f:
        text = "".join(f.readlines())
    return text


def resolveinput(filepath):
    if isfile(filepath):
        return 1


def getcreds(name: str) -> str:
    with open(f"../dotfiles/creds/{name}", "r") as f:
        line = f.readline()
    return line


MODEL = langchain_mistralai.ChatMistralAI(
    api_key=getcreds(name="mistral_bookbot"),
    name=MODEL_ID,
    temperature=0,
)  # , model_kwargs={'rate_limit':(300,10)})


def hashstr(text):
    encoded = text.encode()
    hashed = hashlib.sha1(string=encoded)
    asid = hashed.hexdigest()[:16]
    return asid


def langrequest(agent, messages):
    """@TODO: Swap for longterm fix
    Wrapped the invoke call in some try-except clauses
    because I'm getting rate limited even though I'm on the paid Scale plan,
    and this is a hacky workaround to stall before trying message again
    But yeah, do that asynchronously or something"""
    try:
        try:
            result = agent.invoke(messages)
        except:
            result = agent.invoke(messages)
    except:
        result = None
    return result


def msgs_from_text_arbitrary(prompt: str, chunk_size: int, text: str) -> dict:
    chunks = [text[i : i + MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]
    out = {
        "messages": [
            {"role": "user", "content": f"{prompt} {chunk}"} for chunk in chunks
        ]
    }
    return out


def msgs_from_text(prompt: str, text: str, sep) -> dict:
    chunks = text.split(sep)
    chunks = [chunks[0]] + [f"{sep}{chunk}" for chunk in chunks[1:]]
    out = [{"role": "user", "content": f"{prompt} {chunk}"} for chunk in chunks]
    for msg in out:
        assert len(msg["content"]) < 100000, (
            f"\
        found msg with length {len(msg['content'])}"
        )
    return out


def writeprompt(datatype: str, sep) -> str:
    assert datatype.upper() in OUTPUT_TYPES.keys(), f"\
    Expecting a datatype available in the fields module ({OUTPUT_TYPES.keys()}). Found {
        datatype
    }"
    prompt = (
        f"**Extract information about each {datatype} mentioned in the below text{sep}"
    )
    return prompt


def setupagent(prompt: str, datatype: str, model=MODEL):
    agent = create_agent(
        model=model,
        response_format=OUTPUT_TYPES[datatype.upper()],
        system_prompt=prompt,
    )
    return agent


def process(agent, prompt: str, data: pd.DataFrame) -> pd.DataFrame:
    shuffled = data.copy().sample(data.shape[0])
    responses = []
    for record in shuffled.head(5).itertuples():
        rowid = record.Index
        fname = record.filename
        text = record.text
        msgs = msgs_from_text(prompt=prompt, text=text, sep=TEXT_DELIMITER)
        nmessages = len(msgs)
        logger.info(
            f"processing record ID {rowid} with filename {fname} and {nmessages} messages"
        )
        for i in range(len(msgs)):
            logger.info(f"processing message {i + 1} out of {nmessages}")
            msg = msgs[i]
            resp = langrequest(agent=agent, messages={"messages": msg})
            resp_data = pd.DataFrame([resp["structured_response"].model_dump()])
            resp_data["recordid"] = rowid
            resp_data["filename"] = fname
            resp_data["prompt"] = prompt
            resp_data["section"] = msg["content"].split(PROMPT_SEPARATOR)[1]
            resp_data["section_id"] = resp_data.section.apply(hashstr)
            responses.append(resp_data)
    out = pd.concat(responses)
    return out


# }}}

# --- main --- {{{
if __name__ == "__main__":
    args = getargs()
    setuplogging("logs/llm.log")

    logger.info(f"setting up model and agent for extracting {args.datatype} data")
    prompt = writeprompt(datatype=args.datatype, sep=PROMPT_SEPARATOR)
    agent = setupagent(model=MODEL, prompt=prompt, datatype=args.datatype)

    logger.info("setting up data to process")
    files = getfiles(arg=args.input, fext="txt")
    texts = [{"filename": file.stem, "text": readtext(fname=file)} for file in files]
    texts = pd.DataFrame(texts)

    logger.info("begin processing message(s)")
    responses = process(agent=agent, prompt=prompt, data=texts)

    logger.info("writing formatted response data")
    responses.to_parquet(args.output)
    texts.to_parquet(f"{OUTPUT_DIR}/texts.parquet")

    logger.info("done")
# }}}
