"""
vector_store.py — Módulo de almacenamiento vectorial para el sistema RAG.

Encapsula ChromaDB con embeddings de Sentence Transformers para indexar
y buscar documentos por similitud semántica.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import Optional


class VectorStore:
    """
    Vector store basado en ChromaDB con embeddings de Sentence Transformers.

    Usa el modelo 'all-MiniLM-L6-v2' por defecto: rápido, ligero (~80MB),
    y con buen rendimiento en tareas de similitud semántica.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_COLLECTION = "rag_documents"

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ):
        """
        Args:
            persist_directory: Ruta donde ChromaDB persiste los datos.
            collection_name: Nombre de la colección dentro del store.
            model_name: Nombre del modelo de Sentence Transformers para embeddings.
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.collection_name = collection_name

        # Inicializar ChromaDB con persistencia en disco
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Configurar función de embedding con Sentence Transformers
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # Obtener o crear la colección
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # Similitud coseno
        )

    def add_documents(self, chunks: list[dict]) -> int:
        """
        Añade chunks al vector store.

        Args:
            chunks: Lista de dicts con 'chunk_id', 'text', y 'metadata'.

        Returns:
            Número de chunks añadidos exitosamente.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]

            # Aplanar metadatos para ChromaDB (solo acepta str, int, float, bool)
            flat_meta = self._flatten_metadata(chunk.get("metadata", {}))

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(flat_meta)

        # ChromaDB maneja duplicados silenciosamente (upsert por ID)
        batch_size = 100  # Procesar en lotes para evitar OOM con datasets grandes
        added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            added += len(batch_ids)

        return added

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Busca los chunks más similares a la query.

        Args:
            query_text: Texto de la consulta del usuario.
            top_k: Número máximo de resultados a retornar.
            where: Filtro opcional de metadatos (formato ChromaDB where clause).

        Returns:
            Lista de dicts con 'text', 'metadata', 'distance', y 'similarity'.
        """
        query_params = {
            "query_texts": [query_text],
            "n_results": min(top_k, self.collection.count()),
        }
        if where:
            query_params["where"] = where

        if self.collection.count() == 0:
            return []

        results = self.collection.query(**query_params)

        # Estructurar los resultados
        formatted = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            # ChromaDB con cosine devuelve distancia, convertir a similitud
            similarity = 1.0 - distance

            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": distance,
                "similarity": round(similarity, 4),
            })

        return formatted

    def get_collection_stats(self) -> dict:
        """Retorna estadísticas de la colección actual."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_model": self.model_name,
            "persist_directory": self.persist_directory,
        }

    def delete_collection(self) -> None:
        """Elimina la colección completa y todos sus datos."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _flatten_metadata(self, metadata: dict, prefix: str = "") -> dict:
        """
        Aplana un diccionario anidado para compatibilidad con ChromaDB.
        ChromaDB solo acepta valores de tipo str, int, float, bool en metadatos.
        """
        flat = {}
        for key, value in metadata.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                flat.update(self._flatten_metadata(value, full_key))
            elif isinstance(value, (str, int, float, bool)):
                flat[full_key] = value
            elif isinstance(value, list):
                flat[full_key] = str(value)
            else:
                flat[full_key] = str(value)
        return flat
