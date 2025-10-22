"""RAG pipeline components for PDF processing, summarisation and Q&A.

This module follows the structure described in the Towards AI article about
building a document summarisation and QA workflow without relying on dedicated
RAG frameworks.  It intentionally keeps the code lightweight so it can be
reused both as executable snippets and as material for presentations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence
import base64
import uuid

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


def split_text(text: str, max_len: int = 512, overlap: int = 40) -> List[str]:
    """Split *text* into overlapping chunks.

    Parameters
    ----------
    text:
        Input string extracted from a PDF page.
    max_len:
        Maximum length in characters for each chunk.
    overlap:
        Amount of overlap (in characters) to keep between consecutive chunks.

    Returns
    -------
    list of str
        Ordered list of chunks that preserves reading flow.
    """

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.rstrip(".") + "."
        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}" if current else sentence
        if len(candidate) <= max_len:
            current = candidate
            continue

        chunks.append(current.strip())
        if overlap and len(current) > overlap:
            prefix = current[-overlap:]
            current = f"{prefix} {sentence}".strip()
        else:
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


def _encode_image_to_base64(pixmap: fitz.Pixmap) -> str:
    """Encode a PyMuPDF Pixmap to a base64 PNG string."""

    image_bytes = pixmap.tobytes("png")
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_pdf_chunks(
    pdf_bytes: bytes,
    max_len: int = 512,
    include_images: bool = False,
) -> List[dict]:
    """Extract textual (and optionally visual) chunks from a PDF document."""

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks: List[dict] = []

    try:
        for page_number in range(len(document)):
            page = document.load_page(page_number)
            text = page.get_text("text", sort=True) or ""

            for index, chunk in enumerate(split_text(text, max_len=max_len)):
                chunk_id = f"text_{page_number}_{index}_{uuid.uuid4().hex[:8]}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "type": "text",
                        "page": page_number,
                        "content": chunk,
                        "meta": {"source_page": page_number + 1},
                    }
                )

            if not include_images:
                continue

            for image_index, image in enumerate(page.get_images(full=True)):
                xref = image[0]
                pixmap = fitz.Pixmap(document, xref)
                try:
                    if pixmap.n - pixmap.alpha >= 4:
                        continue

                    caption = f"[image on page {page_number + 1}, idx {image_index}]"
                    image_payload = _encode_image_to_base64(pixmap)
                    chunk_id = f"img_{page_number}_{image_index}_{uuid.uuid4().hex[:8]}"
                    chunks.append(
                        {
                            "id": chunk_id,
                            "type": "image",
                            "page": page_number,
                            "content": caption,
                            "meta": {
                                "source_page": page_number + 1,
                                "image_index": image_index,
                                "image_base64": image_payload,
                            },
                        }
                    )
                finally:
                    pixmap = None
    finally:
        document.close()

    return chunks


LLMCallable = Callable[[str, str], str]


def summarize_chunks(
    chunks: Sequence[dict],
    llm_complete: LLMCallable,
    mode: str = "brief",
) -> tuple[list[str], str]:
    """Summarise a list of chunks and return per-chunk and global summaries."""

    system_prompt = "Você é um assistente que resume artigos técnicos com precisão."
    detailed = mode == "detailed"
    user_template = (
        "Resuma o trecho a seguir, destacando objetivos, achados, método e limitações, se houver:\n\n{texto}"
        if detailed
        else "Resuma de forma concisa o trecho a seguir, focando nas ideias principais:\n\n{texto}"
    )

    per_chunk_summaries: List[str] = []
    for chunk in chunks:
        if chunk.get("type") != "text":
            continue
        user_prompt = user_template.format(texto=chunk["content"])
        per_chunk_summaries.append(llm_complete(system_prompt, user_prompt))

    compose_template = (
        "Integre os resumos abaixo em um ÚNICO resumo estruturado e fiel:\n\n{conteudo}"
        if detailed
        else "Faça um resumo executivo (bulletless) a partir dos pontos abaixo:\n\n{conteudo}"
    )

    combined = "\n\n".join(
        f"Seção {index + 1}: {summary}"
        for index, summary in enumerate(per_chunk_summaries)
    )
    final_summary = llm_complete(system_prompt, compose_template.format(conteudo=combined))

    return per_chunk_summaries, final_summary


class VectorStore:
    """Persistent ChromaDB wrapper that stores PDF chunks."""

    def __init__(
        self,
        persist_dir: str = "./chroma",
        collection: str = "papers",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.embedder = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            ),
        )

        try:
            self.collection = self.client.get_collection(name=collection)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection,
                embedding_function=self._embedding_function,
                metadata={"description": "chunks de PDFs"},
            )

    def _embedding_function(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(list(texts))
        return embeddings.tolist()

    def add_chunks(self, chunks: Iterable[dict]) -> None:
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for chunk in chunks:
            ids.append(chunk["id"])
            documents.append(chunk["content"])
            metadata = {"type": chunk.get("type"), "page": chunk.get("page")}
            metadata.update(chunk.get("meta") or {})
            metadatas.append(metadata)

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, k: int = 4) -> dict:
        embedding = self.embedder.encode(query_text).tolist()
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )


def answer(query: str, vector_store: VectorStore, llm_complete: LLMCallable) -> tuple[str, dict]:
    """Run a grounded QA step using retrieved chunks and an LLM."""

    search_result = vector_store.query(query, k=4)
    documents = search_result["documents"][0]
    system_prompt = "Você responde com base ESTRITA nos trechos fornecidos."
    user_prompt = (
        "Pergunta: {question}\n\nContexto (trechos relevantes):\n{context}\n\n"
        "Responda de forma direta e cite páginas entre parênteses quando possível."
    ).format(question=query, context="\n---\n".join(documents))

    response = llm_complete(system_prompt, user_prompt)
    return response, search_result


@dataclass
class DemoResult:
    """Utility container for presenting the pipeline output."""

    chunk_count: int
    per_chunk_summaries: List[str]
    final_summary: str
    answer: str
    retrieval: dict


def run_demo(pdf_path: str, llm_complete: LLMCallable) -> DemoResult:
    """Execute the full pipeline on *pdf_path* for demonstration purposes."""

    with open(pdf_path, "rb") as handle:
        pdf_bytes = handle.read()

    chunks = extract_pdf_chunks(pdf_bytes)
    per_chunk_summaries, final_summary = summarize_chunks(chunks, llm_complete)

    vector_store = VectorStore()
    vector_store.add_chunks(chunks)

    question = "Quais são os objetivos do trabalho?"
    answer_text, retrieval = answer(question, vector_store, llm_complete)

    return DemoResult(
        chunk_count=len(chunks),
        per_chunk_summaries=per_chunk_summaries,
        final_summary=final_summary,
        answer=answer_text,
        retrieval=retrieval,
    )


__all__ = [
    "split_text",
    "extract_pdf_chunks",
    "summarize_chunks",
    "VectorStore",
    "answer",
    "DemoResult",
    "run_demo",
]
