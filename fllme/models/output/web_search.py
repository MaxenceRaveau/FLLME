from pydantic import BaseModel


class WebSource(BaseModel):
    url: str
    title: str | None = None
    domain: str | None = None


class WebSearchMetadata(BaseModel):
    queries: list[str] = []
    sources: list[WebSource] = []
    rendered_content: str | None = None
