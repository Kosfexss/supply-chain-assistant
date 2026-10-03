"""Local document ingestion and TF-IDF retrieval for the RAG layer."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .database import Database
from .models import Document, RetrievedDocument

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SUPPORTED_SUFFIXES = {".json", ".md", ".txt", ".csv"}


class LocalDocumentStore:
    """A small local retriever; replaceable with a hosted vector store later."""

    def __init__(self, database: Database | None = None) -> None:
        self.database = database
        self._documents: dict[str, Document] = {}

    @property
    def documents(self) -> tuple[Document, ...]:
        if self.database is not None:
            return self.database.list_documents()
        return tuple(self._documents.values())

    def add(self, document: Document) -> None:
        if self.database is not None:
            self.database.save_document(document)
            return
        self._documents[document.document_id] = document

    def ingest_file(self, path: str | Path) -> int:
        """Ingest JSON document records or plain text, Markdown, and CSV files."""
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES:
            return 0

        if suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else [payload]
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object in {file_path}")
                self.add(self._from_record(record, file_path, index))
            return len(records)

        content = file_path.read_text(encoding="utf-8")
        self.add(
            Document(
                document_id=str(file_path.resolve()),
                source=file_path.name,
                category="compliance" if suffix == ".md" else "historical_record",
                content=content,
            )
        )
        return 1

    def ingest_directory(self, path: str | Path) -> int:
        directory = Path(path)
        return sum(
            self.ingest_file(file_path)
            for file_path in sorted(directory.rglob("*"))
            if file_path.is_file() and file_path.suffix.lower() in _SUPPORTED_SUFFIXES
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Return cosine TF-IDF matches, optionally narrowed by metadata."""
        all_documents = self.documents
        if top_k <= 0 or not all_documents:
            return []

        documents = [
            document
            for document in all_documents
            if (category is None or document.category == category)
            and self._matches_filters(document, filters or {})
        ]
        query_terms = self._tokens(query)
        if not documents or not query_terms:
            return []

        tokenized = [self._tokens(document.content) for document in documents]
        document_frequency: Counter[str] = Counter()
        for terms in tokenized:
            document_frequency.update(set(terms))
        total_documents = len(documents)
        query_counts = Counter(query_terms)

        scored: list[RetrievedDocument] = []
        for document, terms in zip(documents, tokenized):
            term_counts = Counter(terms)
            shared_terms = set(query_counts).intersection(term_counts)
            if not shared_terms:
                continue

            query_vector = {
                term: count * self._idf(document_frequency[term], total_documents)
                for term, count in query_counts.items()
            }
            document_vector = {
                term: count * self._idf(document_frequency[term], total_documents)
                for term, count in term_counts.items()
            }
            dot_product = sum(
                query_vector[term] * document_vector[term] for term in shared_terms
            )
            query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
            document_norm = math.sqrt(
                sum(weight * weight for weight in document_vector.values())
            )
            score = dot_product / (query_norm * document_norm)
            scored.append(
                RetrievedDocument(
                    document=document,
                    score=score,
                    excerpt=self._excerpt(document.content, shared_terms),
                )
            )

        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

    @staticmethod
    def _from_record(record: dict[str, Any], path: Path, index: int) -> Document:
        content = record.get("content", "")
        metadata = record.get("metadata", {})
        if not isinstance(content, str) or not isinstance(metadata, dict):
            raise ValueError(f"Invalid document content or metadata in {path}")
        return Document(
            document_id=str(record.get("document_id", f"{path.resolve()}#{index}")),
            source=str(record.get("source", path.name)),
            category=str(record.get("category", "historical_record")),
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _matches_filters(document: Document, filters: dict[str, Any]) -> bool:
        return all(document.metadata.get(key) == value for key, value in filters.items())

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _idf(document_frequency: int, total_documents: int) -> float:
        return math.log((1 + total_documents) / (1 + document_frequency)) + 1

    @staticmethod
    def _excerpt(content: str, terms: Iterable[str], max_length: int = 240) -> str:
        normalized = " ".join(content.split())
        lowered = normalized.lower()
        positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
        if not positions or len(normalized) <= max_length:
            return normalized[:max_length]
        start = max(0, min(positions) - max_length // 4)
        return normalized[start : start + max_length]