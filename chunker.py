"""
chunker.py — Módulo de fragmentación de texto para el sistema RAG.
Actualizado para usar RecursiveCharacterTextSplitter de Langchain.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

class TextChunker:
    """
    Divide texto en fragmentos estructuralmente consistentes usando
    RecursiveCharacterTextSplitter de Langchain.
    """
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        strategy: str = "sentence",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        
        # Ignoramos la "strategy" antigua ingenua y usamos RecursiveCharacterTextSplitter
        # que es estructuralmente consciente para respetar párrafos y Markdown.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def _generate_chunk_id(self, text: str, source: str, index: int) -> str:
        raw = f"{source}::{index}::{text[:50]}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def chunk_text(self, text: str, source: str = "unknown") -> list[dict]:
        if not text or not text.strip():
            return []

        # Langchain hace el particionado seguro y robusto
        raw_chunks = self.splitter.split_text(text)

        chunks = []
        char_offset = 0

        for i, chunk_text in enumerate(raw_chunks):
            start_pos = text.find(chunk_text[:50], max(0, char_offset - self.overlap))
            if start_pos == -1:
                start_pos = char_offset

            chunk_id = self._generate_chunk_id(chunk_text, source, i)

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "start_char": start_pos,
                    "end_char": start_pos + len(chunk_text),
                    "char_count": len(chunk_text),
                },
            })
            char_offset = start_pos + len(chunk_text) - self.overlap

        return chunks

    def chunk_document(self, document: dict) -> list[dict]:
        source = document.get("metadata", {}).get("filename", "unknown")
        chunks = self.chunk_text(document["content"], source=source)
        for chunk in chunks:
            chunk["metadata"]["original_metadata"] = document.get("metadata", {})
        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
