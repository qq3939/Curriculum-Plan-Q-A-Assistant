from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from .embeddings import EmbeddingProvider, normalize_matrix


INDEX_VERSION = 1


@dataclass
class PdfChunk:
    chunk_id: str
    source_file: str
    page: int
    section_title: str
    text: str


@dataclass
class IndexBundle:
    chunks: list[PdfChunk]
    embeddings: np.ndarray
    manifest: dict[str, Any]


def get_pdf_files(corpus_dir: Path) -> list[Path]:
    return sorted(corpus_dir.glob("*.pdf"), key=lambda path: path.name)


def build_index(corpus_dir: Path, index_dir: Path, provider: EmbeddingProvider) -> IndexBundle:
    pdf_files = get_pdf_files(corpus_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under {corpus_dir}")

    chunks: list[PdfChunk] = []
    files_meta: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        sections = _section_titles_by_page(doc, pdf_path.stem)
        stat = pdf_path.stat()
        files_meta.append(
            {
                "name": pdf_path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "pages": doc.page_count,
            }
        )

        for page_index in range(doc.page_count):
            page_no = page_index + 1
            text = _clean_text(doc.load_page(page_index).get_text("text"))
            if not text:
                continue
            for part_index, part in enumerate(_split_text(text)):
                chunks.append(
                    PdfChunk(
                        chunk_id=f"{pdf_path.stem}:p{page_no}:c{part_index + 1}",
                        source_file=pdf_path.name,
                        page=page_no,
                        section_title=sections.get(page_no, pdf_path.stem),
                        text=part,
                    )
                )

    texts = [chunk.text for chunk in chunks]
    embeddings = provider.embed_texts(texts)
    embeddings = normalize_matrix(embeddings)

    manifest = {
        "version": INDEX_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "embedding_signature": provider.signature,
        "file_count": len(files_meta),
        "chunk_count": len(chunks),
        "files": files_meta,
    }

    save_index(index_dir, chunks, embeddings, manifest)
    return IndexBundle(chunks=chunks, embeddings=embeddings, manifest=manifest)


def save_index(index_dir: Path, chunks: list[PdfChunk], embeddings: np.ndarray, manifest: dict[str, Any]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    np.save(index_dir / "embeddings.npy", embeddings.astype(np.float32))


def load_index(index_dir: Path) -> IndexBundle:
    manifest_path = index_dir / "manifest.json"
    chunks_path = index_dir / "chunks.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    if not manifest_path.exists() or not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError("Index files are missing.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: list[PdfChunk] = []
    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(PdfChunk(**json.loads(line)))
    embeddings = normalize_matrix(np.load(embeddings_path))
    if len(chunks) != len(embeddings):
        raise ValueError("Index chunk count does not match embedding count.")
    return IndexBundle(chunks=chunks, embeddings=embeddings, manifest=manifest)


def is_index_current(corpus_dir: Path, index_dir: Path, provider_signature: str) -> bool:
    try:
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("version") != INDEX_VERSION:
        return False
    if manifest.get("embedding_signature") != provider_signature:
        return False
    if not (index_dir / "chunks.jsonl").exists() or not (index_dir / "embeddings.npy").exists():
        return False

    current_files = _current_file_meta(corpus_dir)
    saved_files = [
        {
            "name": item.get("name"),
            "size": item.get("size"),
            "mtime_ns": item.get("mtime_ns"),
        }
        for item in manifest.get("files", [])
    ]
    return current_files == saved_files


def _current_file_meta(corpus_dir: Path) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []
    for pdf_path in get_pdf_files(corpus_dir):
        stat = pdf_path.stat()
        meta.append({"name": pdf_path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return meta


def _section_titles_by_page(doc: fitz.Document, fallback_title: str) -> dict[int, str]:
    toc = doc.get_toc(simple=True)
    if not toc:
        return {page: fallback_title for page in range(1, doc.page_count + 1)}

    entries: list[tuple[int, str]] = []
    stack: list[str] = []
    for level, title, page in toc:
        level = max(level, 1)
        stack = stack[: level - 1]
        stack.append(_compact_title(title))
        entries.append((max(int(page), 1), " > ".join(item for item in stack if item)))

    entries.sort(key=lambda item: item[0])
    sections: dict[int, str] = {}
    entry_index = 0
    current = entries[0][1] if entries else fallback_title
    for page_no in range(1, doc.page_count + 1):
        while entry_index < len(entries) and entries[entry_index][0] <= page_no:
            current = entries[entry_index][1]
            entry_index += 1
        sections[page_no] = current or fallback_title
    return sections


def _compact_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


def _clean_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _split_text(text: str, max_chars: int = 1600, overlap_lines: int = 6) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines()
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        add_len = len(line) + 1
        if current and current_len + add_len > max_chars:
            parts.append("\n".join(current).strip())
            current = current[-overlap_lines:] if overlap_lines else []
            current_len = sum(len(item) + 1 for item in current)
        current.append(line)
        current_len += add_len
    if current:
        parts.append("\n".join(current).strip())
    return [part for part in parts if part]
