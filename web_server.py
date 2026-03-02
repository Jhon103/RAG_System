"""
web_server.py — Servidor web FastAPI asíncrono para el sistema RAG.
Reescrito para usar FastAPI y Uvicorn, soportando asincronía y previniendo bloqueos concurrentes.
"""

import os
import sys
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_loader import DocumentLoader
from chunker import TextChunker
from vector_store import VectorStore
from retriever import Retriever
from generator import ResponseGenerator

# ─── Configuración ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
STATIC_DIR = os.path.join(BASE_DIR, "static")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNK_STRATEGY = "sentence"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.15

# Inicialización segura (Singleton de componentes)
rag_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Inicializando componentes RAG (FastAPI)...")
    vector_store = VectorStore(persist_directory=CHROMA_DIR, collection_name="rag_documents")
    retriever = Retriever(vector_store=vector_store, top_k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD)
    # Instanciamos el generador que ahora usará Ollama por defecto
    generator = ResponseGenerator(max_context_chunks=TOP_K)
    
    rag_components["vector_store"] = vector_store
    rag_components["retriever"] = retriever
    rag_components["generator"] = generator
    print("[✓] Componentes RAG listos y expuestos.")
    yield
    print("[*] Cerrando componentes RAG de forma segura...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/api/stats")
async def stats():
    try:
        vs = rag_components["vector_store"]
        stats_data = await run_in_threadpool(vs.get_collection_stats)
        stats_data["config"] = {
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "strategy": CHUNK_STRATEGY,
            "top_k": TOP_K,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }
        return stats_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files():
    try:
        if not os.path.exists(DOCS_DIR):
            return {"files": [], "total": 0}

        loader = DocumentLoader(DOCS_DIR)
        files = loader.get_supported_files()
        file_list = []
        for f in files:
            full_path = os.path.join(DOCS_DIR, f)
            size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
            ext = os.path.splitext(f)[1]
            file_list.append({"name": f, "size_bytes": size, "extension": ext})

        return {"files": file_list, "total": len(file_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query(req: QueryRequest):
    try:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="El campo 'query' es requerido.")

        vs = rag_components["vector_store"]
        ret = rag_components["retriever"]
        gen = rag_components["generator"]

        # Validar si hay chunks de forma no bloqueante
        stats = await run_in_threadpool(vs.get_collection_stats)
        if stats["total_chunks"] == 0:
            return {
                "error": "No hay documentos indexados. Por favor indexa primero.",
                "answer": None,
                "results": [],
            }

        start = time.time()
        # Recuperación
        context = await run_in_threadpool(ret.retrieve_with_context, req.query, top_k=req.top_k)
        
        # Inferencia LLM delegada a ThreadPool para no bloquear el Event Loop
        response = await run_in_threadpool(
            gen.generate,
            query=req.query,
            retrieval_results=context["results"],
            include_sources=True,
            include_scores=True,
        )
        elapsed = time.time() - start

        results_json = []
        for r in context["results"]:
            results_json.append({
                "text": r["text"],
                "source": r.get("metadata", {}).get("source", "unknown"),
                "similarity": r.get("similarity", 0),
                "rank": r.get("rank", 0),
                "chunk_index": r.get("metadata", {}).get("chunk_index", 0),
            })

        return {
            "query": req.query,
            "answer": response["answer"],
            "confidence": response["confidence"],
            "chunks_used": response["chunks_used"],
            "results": results_json,
            "sources": list({r["source"] for r in results_json}),
            "avg_similarity": context["avg_similarity"],
            "elapsed_ms": round(elapsed * 1000, 1),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index")
async def index_docs():
    try:
        if not os.path.exists(DOCS_DIR):
            raise HTTPException(status_code=400, detail="El directorio ./docs/ no existe.")

        loader = DocumentLoader(DOCS_DIR)
        documents = loader.load_directory(recursive=True)

        if not documents:
            raise HTTPException(status_code=400, detail="No se encontraron documentos.")

        chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            strategy=CHUNK_STRATEGY,
        )
        chunks = chunker.chunk_documents(documents)
        
        vs = rag_components["vector_store"]
        added = await run_in_threadpool(vs.add_documents, chunks)

        file_details = []
        for doc in documents:
            meta = doc.get("metadata", {})
            file_details.append({
                "filename": meta.get("filename", "unknown"),
                "size_bytes": meta.get("size_bytes", 0),
                "extension": meta.get("extension", ""),
            })

        return {
            "status": "ok",
            "documents_loaded": len(documents),
            "chunks_generated": len(chunks),
            "chunks_indexed": added,
            "files": file_details,
            "config": {
                "chunk_size": CHUNK_SIZE,
                "overlap": CHUNK_OVERLAP,
                "strategy": CHUNK_STRATEGY,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_db():
    try:
        vs = rag_components["vector_store"]
        await run_in_threadpool(vs.delete_collection)
        return {"status": "ok", "message": "Base de datos vectorial limpiada exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Montar frontend estático
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor Uvicorn FastAPI...")
    uvicorn.run("web_server:app", host="0.0.0.0", port=8080, reload=False)
