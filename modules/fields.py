#!/usr/bin/env python3
# vim: set ts=4 sts=0 sw=4 si fenc=utf-8 et:
# vim: set fdm=marker fmr={{{,}}} fdl=0 foldcolumn=4:
# Authors:     BP
# Maintainers: BP
# =========================================

# ---- dependencies {{{
from datetime import datetime
from pydantic import BaseModel, Field
# }}}


# --- support methods --- {{{
class Contact(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")


class Mention(BaseModel):
    """Information about a person named in a text, \
    including some personal and contextual details."""
    name: str = Field(description="The name of the person mentioned in the text.")
    email: str = Field(description="The email address of the person.")
    phone: str = Field(description="The phone number of the person.")
    context: str = Field(description="The context in which the name was mentioned.")
    mention: str = Field(description="How the person is named and/or mentioned in the text.")


class Event(BaseModel):
    """Information about an event described in a text,
    including date, time, location, person(s) involved, context,
    and citation (a snippet of the exact text mentioning event).
    At extraction, there is no minimum number/combination of criteria
    necessary to capture an event."""
    timestamp: datetime = Field(description="The calendar date and time of the event.")
    location: str = Field(description="The geographic location of the event (as plain text).")
    names: list[str] = Field(description="The name of any person(s) mentioned in the text.")
    category: str = Field(description="A categorical label for the event described in the text.")
    summary: str = Field(description="A summary of the event described in the text.")
    citation: str = Field(description="A snippet of the exact text mentioning event.")


class AllegationInfo(BaseModel):
    """Contact information for a person."""
    dates: list[dict] = Field(description="A dictionary with every date as a key and \
    the corresponding value is what occurred on that date.")
    allegations: str = Field(description="The summary of the allegation(s) \
    as presented in the text.")
    conduct_category: str = Field(description="The category of conduct \
    associated with the allegation(s).")
    finding_code: str = Field(description="The short code referring to \
    the category of the finding.")
    dept_action: str = Field(description="Any department action mentioned \
    in reference to the allegation(s).")
    findings: str = Field(description="A summary of the findings of fact \
    describing the agency's investigation.")
# }}}