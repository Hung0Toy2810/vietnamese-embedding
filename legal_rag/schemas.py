"""Persistent schemas for source spans, semantic graphs and chunk plans."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Span(Record):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    page: int | None = Field(default=None, ge=1)
    locator: str = ""


class ParsedDocument(Record):
    text: str
    spans: list[Span]
    pages: int = 0
    warnings: list[str] = Field(default_factory=list)
    parser: str


class Atom(Record):
    id: str
    text: str
    start: int
    end: int
    spans: list[Span]


class ChunkPlan(Record):
    rationale: str = Field(min_length=1, max_length=1600)
    topics: list[str] = Field(default_factory=list, max_length=16)
    similarity_quantile: float = Field(default=0.65, ge=0.0, le=1.0)
    parent_token_limit: int = Field(default=1200, ge=256, le=3000)
    child_token_limit: int = Field(default=224, ge=48, le=224)
    verify_relations: Literal[True] = True
    coverage_notes: list[str] = Field(default_factory=list, max_length=12)


class Relation(Record):
    source: str
    target: str
    related: bool = Field(strict=True)
    kind: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=160)


class Relations(Record):
    relations: list[Relation] = Field(default_factory=list, max_length=48)


class Subject(Record):
    anchor: str
    name: str = Field(min_length=1, max_length=120)


class Membership(Record):
    unit: str
    subject: str | None
    relation: str = Field(min_length=1, max_length=80)

    @field_validator("subject", mode="before")
    @classmethod
    def normalize_null(cls, value):
        # Some local models encode JSON null as the literal string "null".
        return None if value == "null" else value


class ScopeGraph(Record):
    subjects: list[Subject]
    members: list[Membership]


class Subjects(Record):
    subjects: list[Subject]


class Memberships(Record):
    members: list[Membership]


class SourceReference(Record):
    source: int = Field(ge=1, le=1000, strict=True)
    passage: int = Field(ge=1, le=1000, strict=True)


class Evidence(SourceReference):
    quote: str = Field(min_length=1)


class Answer(Record):
    answerable: bool = Field(strict=True)
    answer: str
    evidence: list[Evidence] = Field(default_factory=list, max_length=30)
    follow_up: list[str] = Field(default_factory=list, max_length=3)


class Job(Record):
    id: str
    document_id: str
    kind: Literal["ingest", "answer"]
    status: Literal["queued", "running", "done", "error"] = "queued"
    message: str = "Queued"
    payload: dict = Field(default_factory=dict)
    result: dict | None = None
    created: float
