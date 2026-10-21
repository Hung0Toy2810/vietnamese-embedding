"""Generator-tokenizer context estimates with explicit safety headroom."""

import json
from functools import lru_cache

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from legal_rag.settings import Settings


class ContextBudgetExceeded(ValueError):
    """A completed-turn boundary would exceed the configured input budget."""


@lru_cache(maxsize=2)
def generator_tokenizer(repository: str, revision: str) -> Tokenizer:
    path = hf_hub_download(repository, "tokenizer.json", revision=revision)
    return Tokenizer.from_file(path)


def token_count(text: str, settings: Settings) -> int:
    tokenizer = generator_tokenizer(
        settings.generator_tokenizer, settings.generator_tokenizer_revision
    )
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def ensure_context(messages: list[dict], tools: list[dict], settings: Settings) -> int:
    """Count serialized content, reserve tool/chat overhead and generation space.

    Ollama templates can differ from Hugging Face templates. The tokenizer count
    is therefore accompanied by a conservative overhead allowance, not advertised
    as an exact server-side token count.
    """
    serialized = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
    estimated = token_count(serialized, settings) + 512 + 64 * len(messages)
    reserve = max(2048, settings.context_tokens // 4)
    if estimated + reserve > settings.context_tokens:
        raise ContextBudgetExceeded(
            f"Context requires about {estimated} input tokens plus {reserve} reserved tokens; configured limit is {settings.context_tokens}."
        )
    return estimated
