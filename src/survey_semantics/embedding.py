"""Text embedding backends for item wording."""

import os
import re
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    backend: str
    model_name: str
    requested_backend: str
    requested_model_name: str
    slug: str


OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

_SOCKET_BLOCK_INSTALLED = False


def embed_texts(
    texts: Sequence[str],
    method: str = "auto",
    model_name: Optional[str] = None,
) -> np.ndarray:
    """Embed item text with an optional transformer backend and TF-IDF fallback."""

    return embed_texts_with_metadata(texts, method=method, model_name=model_name).vectors


def embed_texts_with_metadata(
    texts: Sequence[str],
    method: str = "auto",
    model_name: Optional[str] = None,
) -> EmbeddingResult:
    """Embed item text and return provenance for output naming."""

    method = (method or "auto").lower()
    requested_model = model_name or ""
    if method in {"auto", "sentence-transformers", "sentence_transformers"}:
        try:
            resolved_model = model_name or "BAAI/bge-m3"
            vectors = _sentence_transformer_embeddings(texts, resolved_model)
            backend = "sentence-transformers"
            return EmbeddingResult(
                vectors=vectors,
                backend=backend,
                model_name=resolved_model,
                requested_backend=method,
                requested_model_name=requested_model,
                slug=embedding_slug(backend, resolved_model),
            )
        except Exception:
            if method != "auto":
                raise
    backend = "tfidf"
    resolved_model = "word-1-2gram-max1024"
    return EmbeddingResult(
        vectors=_tfidf_embeddings(texts),
        backend=backend,
        model_name=resolved_model,
        requested_backend=method,
        requested_model_name=requested_model,
        slug=embedding_slug(backend, resolved_model),
    )


def embedding_slug(backend: str, model_name: Optional[str] = None) -> str:
    label = backend if not model_name else "{}__{}".format(backend, model_name)
    label = label.replace("/", "_").replace("\\", "_")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return label or "embedding"


def _sentence_transformer_embeddings(
    texts: Sequence[str],
    model_name: Optional[str] = None,
) -> np.ndarray:
    enforce_local_ai_offline_policy()
    with block_outbound_sockets():
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(model_name or "BAAI/bge-m3", local_files_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "This sentence-transformers version does not expose local_files_only=True. "
                "Refusing to load the model because the pipeline is configured to prohibit "
                "all Hugging Face/network callbacks."
            ) from exc

        vectors = model.encode(list(texts), normalize_embeddings=True)
    return np.asarray(vectors, dtype=float)


def enforce_local_ai_offline_policy() -> None:
    """Set offline/telemetry-disable environment variables before HF imports."""

    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value


def install_outbound_socket_blocker() -> None:
    """Install a process-wide outbound network blocker."""

    global _SOCKET_BLOCK_INSTALLED
    if _SOCKET_BLOCK_INSTALLED:
        return

    def blocked_connect(self, address):
        raise RuntimeError(
            "Outbound network access is disabled for local survey analysis. "
            "Use local files and locally cached/local embedding models only."
        )

    def blocked_connect_ex(self, address):
        raise RuntimeError(
            "Outbound network access is disabled for local survey analysis. "
            "Use local files and locally cached/local embedding models only."
        )

    def blocked_create_connection(*args, **kwargs):
        raise RuntimeError(
            "Outbound network access is disabled for local survey analysis. "
            "Use local files and locally cached/local embedding models only."
        )

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    socket.create_connection = blocked_create_connection
    _SOCKET_BLOCK_INSTALLED = True


@contextmanager
def block_outbound_sockets():
    """Block outbound socket connections while local embedding code runs."""

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def blocked_connect(self, address):
        raise RuntimeError(
            "Outbound network access is disabled for local survey embedding. "
            "Use a model that is already present on local disk or in the local HF cache."
        )

    def blocked_create_connection(*args, **kwargs):
        raise RuntimeError(
            "Outbound network access is disabled for local survey embedding. "
            "Use a model that is already present on local disk or in the local HF cache."
        )

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect
    socket.create_connection = blocked_create_connection
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create_connection


def _tfidf_embeddings(texts: Sequence[str]) -> np.ndarray:
    cleaned = [text if str(text).strip() else "item" for text in texts]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=1024,
    )
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        vectorizer = TfidfVectorizer(lowercase=True, analyzer="char", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(cleaned)
    dense = matrix.toarray().astype(float)
    return normalize(dense, norm="l2", axis=1)
