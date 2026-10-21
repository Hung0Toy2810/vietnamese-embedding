"""Validated runtime settings; document-specific decisions belong in ChunkPlan."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    root: Path = Path(__file__).resolve().parent / "data" / "workspace"
    model: str = "qwen3.5:9b"
    model_revision: str = ""
    generator_tokenizer: str = "Qwen/Qwen3.5-9B"
    generator_tokenizer_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    ollama_host: str = "http://localhost:11434"
    device: str = "cpu"
    parser: str = "docling"
    max_upload_bytes: int = 25 * 1024 * 1024
    max_characters: int = 2_000_000
    max_pages: int = 600
    atom_tokens: int = Field(default=180, ge=24, le=224)
    candidate_neighbors: int = Field(default=4, ge=1, le=16)
    relation_batch_size: int = Field(default=8, ge=1, le=24)
    ingestion_thinking: bool = False
    answer_thinking: bool = False
    graph_window_tokens: int = Field(default=2400, ge=512, le=6000)
    candidate_limit: int = Field(default=100, ge=1, le=500)
    rerank_limit: int = Field(default=10, ge=1, le=100)
    context_tokens: int = Field(default=16384, ge=4096)
    source_bytes: int = Field(default=16000, ge=1000, le=64000)
    model_rounds: int = Field(default=8, ge=2, le=30)
    plan_attempts: int = Field(default=2, ge=1, le=4)
    web_domains: tuple[str, ...] = ("gov.vn", "chinhphu.vn", "vbpl.vn")

    @classmethod
    def from_env(cls) -> "Settings":
        values = {}
        for name in cls.model_fields:
            value = os.getenv("RAG_" + name.upper())
            if value is not None:
                values[name] = (
                    tuple(value.split(",")) if name == "web_domains" else value
                )
        return cls.model_validate(values)
