"""
generator.py — Módulo de generación de respuestas para el sistema RAG.
Actualizado para usar la API local de Ollama.
"""

import httpx
import json
from typing import Optional

class ResponseGenerator:
    """
    Generador de respuestas usando Ollama localmente.
    """

    def __init__(self, max_context_chunks: int = 5, model_id: str = "hf.co/LiquidAI/LFM2-2.6B-Exp-GGUF:Q4_K_M"):
        self.max_context_chunks = max_context_chunks
        self.model_id = model_id
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"[*] Configurado para usar Ollama con el modelo: {self.model_id}")

    def generate(
        self,
        query: str,
        retrieval_results: list[dict],
        include_sources: bool = True,
        include_scores: bool = True,
    ) -> dict:
        if not retrieval_results:
            return self._no_results_response(query)

        chunks = retrieval_results[: self.max_context_chunks]
        context_text = "\n\n".join([c["text"].strip() for c in chunks])

        sources_info = []
        for c in chunks:
            sources_info.append({
                "source": c.get("metadata", {}).get("source", "unknown"),
                "chunk_index": c.get("metadata", {}).get("chunk_index", "?"),
                "similarity": c.get("similarity", 0.0),
            })
            
        avg_similarity = (
            sum(c.get("similarity", 0) for c in chunks) / len(chunks)
            if chunks else 0.0
        )
        
        confidence = min(avg_similarity * 1.5, 1.0)

        # Prompt restrictivo para RAG
        prompt = f"""Responde a la pregunta en español utilizando ÚNICAMENTE la información del contexto proporcionado.
Si la respuesta no se encuentra en el contexto, di "No puedo responder a esa pregunta basado en la información proporcionada."
No agregues información externa, no inventes hechos.

Contexto:
{context_text}

Pregunta: {query}
Respuesta:"""

        # Inferencia con Ollama
        try:
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Temperatura baja para mayor apego al contexto
                    "top_p": 0.9
                }
            }
            
            # Usamos httpx.post síncrono ya que está envuelto en run_in_threadpool en fastapi
            response = httpx.post(self.ollama_url, json=payload, timeout=60.0)
            response.raise_for_status()
            result_json = response.json()
            answer_text = result_json.get("response", "").strip()

        except Exception as e:
            answer_text = f"Error al generar respuesta con Ollama: {str(e)}"

        # Agregar fuentes
        if include_sources and sources_info:
            answer_text += "\n\n📚 Fuentes citadas:\n"
            seen = set()
            for src in sources_info:
                if src['source'] not in seen:
                    seen.add(src['source'])
                    score_str = f" (similitud: {src['similarity']:.2%})" if include_scores else ""
                    answer_text += f"  • {src['source']}{score_str}\n"

        return {
            "answer": answer_text.strip(),
            "sources": sources_info,
            "confidence": confidence,
            "chunks_used": len(chunks),
        }

    def _no_results_response(self, query: str) -> dict:
        return {
            "answer": f"❌ No se encontraron resultados relevantes en el contexto para: \"{query}\"",
            "sources": [],
            "confidence": 0.0,
            "chunks_used": 0,
        }
