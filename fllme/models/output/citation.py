from pydantic import BaseModel
from enum import StrEnum


class CitationType(StrEnum):
    URL = "url"
    DOCUMENT = "document"
    SEARCH = "search"
    SEARCH_QUERY = "search_query"


class TextSpan(BaseModel):
    start: int
    end: int
    text: str | None = None


class Citation(BaseModel):
    type: CitationType
    url: str | None = None
    title: str | None = None
    domain: str | None = None
    content: str | None = None
    document_id: str | None = None
    span: TextSpan | None = None
    confidence: float | None = None
