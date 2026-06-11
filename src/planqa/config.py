from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(root_dir: Path | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    if root_dir is not None:
        load_dotenv(root_dir / ".env", encoding="utf-8-sig")
    else:
        load_dotenv(encoding="utf-8-sig")


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    corpus_dir: Path
    index_dir: Path
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    embedding_batch_size: int = 64

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def has_chat_config(self) -> bool:
        return self.has_api_key and bool(self.chat_model.strip())

    @property
    def uses_local_embeddings(self) -> bool:
        return self.embedding_model.strip().lower() in {"", "local", "local-hash", "local_hash"}


def load_config(root_dir: Path | None = None) -> AppConfig:
    root = (root_dir or Path.cwd()).resolve()
    _load_dotenv(root)
    return AppConfig(
        root_dir=root,
        corpus_dir=root / "培养计划",
        index_dir=root / "data" / "index",
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
    )
