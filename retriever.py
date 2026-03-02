"""
retriever.py — Módulo de recuperación semántica para el sistema RAG.

Orquesta la búsqueda de fragmentos relevantes dado un query del usuario,
combinando el vector store con lógica de ranking y filtrado.
"""

from vector_store import VectorStore
from typing import Optional


class Retriever:
    """
    Motor de recuperación semántica.
    
    Dado un query en lenguaje natural, encuentra los chunks más relevantes
    de la base de conocimiento usando búsqueda por similitud vectorial.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.2,
    ):
        """
        Args:
            vector_store: Instancia de VectorStore con los documentos indexados.
            top_k: Número máximo de chunks a recuperar por query.
            similarity_threshold: Umbral mínimo de similitud (0.0-1.0).
                Chunks por debajo de este umbral se descartan.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> list[dict]:
        """
        Recupera los chunks más relevantes para la query.

        Args:
            query: Pregunta o texto de búsqueda del usuario.
            top_k: Override del top_k por defecto para esta query.
            filter_source: Filtrar resultados por nombre de archivo fuente.

        Returns:
            Lista de resultados ordenados por similitud descendente, cada uno con:
            - 'text': Contenido del chunk.
            - 'metadata': Metadatos del chunk.
            - 'similarity': Score de similitud (0.0-1.0).
            - 'rank': Posición en el ranking (1-indexed).
        """
        k = top_k or self.top_k

        # Construir filtro de metadatos para ChromaDB
        where_filter = None
        if filter_source:
            where_filter = {"source": filter_source}

        # Ejecutar búsqueda vectorial
        raw_results = self.vector_store.query(
            query_text=query,
            top_k=k,
            where=where_filter,
        )

        # Filtrar por umbral de similitud y asignar ranking
        filtered = []
        for i, result in enumerate(raw_results):
            if result["similarity"] >= self.similarity_threshold:
                result["rank"] = i + 1
                filtered.append(result)

        return filtered

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Recupera chunks relevantes junto con metadatos de contexto.

        Returns:
            Diccionario con:
            - 'query': El query original.
            - 'results': Lista de chunks recuperados.
            - 'total_found': Número total de resultados encontrados.
            - 'sources': Lista única de fuentes usadas.
            - 'avg_similarity': Similitud promedio de los resultados.
        """
        results = self.retrieve(query, top_k=top_k)

        sources = list({
            r["metadata"].get("source", "unknown")
            for r in results
        })

        avg_sim = (
            sum(r["similarity"] for r in results) / len(results)
            if results
            else 0.0
        )

        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "sources": sources,
            "avg_similarity": round(avg_sim, 4),
        }

    def get_stats(self) -> dict:
        """Retorna estadísticas del retriever y su vector store."""
        store_stats = self.vector_store.get_collection_stats()
        return {
            **store_stats,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
        }
