import os
import sys
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def main():
    doc = Document()
    
    # Título principal
    title = doc.add_heading('Documentación Técnica y Propuesta de Paso a Producción', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading('1. Resumen Ejecutivo', level=1)
    doc.add_paragraph(
        "El presente documento detalla la arquitectura actual del sistema de Generación Aumentada por Recuperación (RAG) "
        "implementado en el proyecto. Se describe el flujo de datos, los componentes modulares (Carga de documentos, "
        "Fragmentación, Almacenamiento Vectorial, Recuperación y Generación) y sus interfaces (CLI y API Web). "
        "Además, se proponen mejoras críticas, metodologías y tecnologías recomendadas para llevar este prototipo "
        "a un entorno de producción altamente disponible, escalable y seguro."
    )
    
    doc.add_heading('2. Arquitectura del Sistema Actual', level=1)
    doc.add_paragraph("El sistema actual está compuesto por los siguientes módulos:")
    
    components = [
        ("DocumentLoader (document_loader.py):", "Encargado de la ingesta de documentos locales (.txt, .md, .pdf, etc.). Lee los archivos y extrae su contenido junto con metadatos útiles (ruta, tamaño, extensión)."),
        ("TextChunker (chunker.py):", "Utiliza RecursiveCharacterTextSplitter de LangChain para dividir los textos largos en fragmentos (chunks) manejables (500 caracteres por defecto con 100 de solapamiento). Esto mantiene la coherencia semántica respetando párrafos y estructuras."),
        ("VectorStore (vector_store.py):", "Implementado sobre ChromaDB local (persist_directory). Transforma los fragmentos de texto en vectores matemáticos (embeddings) usando el modelo ligero 'all-MiniLM-L6-v2' de Sentence Transformers y los indexa mediante similitud coseno."),
        ("Retriever (retriever.py):", "Recibe la consulta del usuario, la vectoriza y busca los K fragmentos más similares en ChromaDB, aplicando un umbral de similitud mínimo para garantizar relevancia."),
        ("ResponseGenerator (generator.py):", "Conecta con una instancia local de Ollama (modelo 'hf.co/LiquidAI/LFM2-2.6B-Exp-GGUF:Q4_K_M') vía HTTP. Construye un prompt restrictivo con el contexto recuperado para mitigar alucinaciones y generar la respuesta final."),
        ("Interfaces de Usuario:", "Cuenta con una Interfaz de Línea de Comandos (main.py) usando Rich para experiencias de terminal, y un Servidor Web (web_server.py) usando FastAPI para exponer endpoints asíncronos RESTful y servir un cliente web estático.")
    ]
    
    for name, desc in components:
        p = doc.add_paragraph()
        p.add_run(name).bold = True
        p.add_run(f" {desc}")
        
    doc.add_heading('3. Flujo de Datos', level=1)
    doc.add_heading('3.1 Ingesta de Datos (Indexing)', level=2)
    doc.add_paragraph("Documentos -> DocumentLoader -> Textos -> TextChunker -> Chunks -> VectorStore (Embeddings) -> ChromaDB")
    
    doc.add_heading('3.2 Generación (Querying)', level=2)
    doc.add_paragraph("Pregunta Usuario -> Retriever -> ChromaDB (Búsqueda por similitud) -> Contexto Recuperado -> ResponseGenerator (Prompt + Contexto) -> Ollama LLM -> Respuesta Final")
    
    doc.add_heading('4. Recomendaciones para el Entorno de Producción', level=1)
    doc.add_paragraph("Aunque el prototipo es funcional, para desplegarlo en un entorno empresarial o de producción con alto tráfico y exigencias de confiabilidad, se deben aplicar las siguientes mejoras:")

    doc.add_heading('4.1 Arquitectura y Escalabilidad', level=2)
    escalabilidad = [
        ("Base de Datos Vectorial:", "Migrar de ChromaDB embebido local a una arquitectura cliente-servidor o Cloud. Alternativas recomendadas: Qdrant, Milvus o Pinecone. Esto permite escalado horizontal, alta disponibilidad y replicación."),
        ("Inferencia LLM:", "La inferencia con Ollama HTTP secuencial se convertirá en un cuello de botella. Se recomienda usar motores de inferencia optimizados para producción como vLLM o TGI (Text Generation Inference) de HuggingFace, que soportan Continuous Batching y PagedAttention para manejar múltiples requests simultáneos."),
        ("Procesamiento Asíncrono (Colas):", "La indexación de documentos (ruta /api/index) es intensiva. En producción, la ingesta debe encolarse usando Celery o RabbitMQ/Redis. FastAPI solo debería recibir el archivo y retornar un 'Job ID' mientras un worker en background procesa el chunking y embedding.")
    ]
    for item, desc in escalabilidad:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(item).bold = True
        p.add_run(f" {desc}")

    doc.add_heading('4.2 Calidad del RAG (Mejores Métodos)', level=2)
    rag_quality = [
        ("Estrategias de Chunking Avanzadas:", "Implementar 'Semantic Chunking' o 'Parent-Child Retriever'. Esto implica guardar fragmentos pequeños en el índice vectorial para mayor precisión en la búsqueda, pero enviando al LLM el bloque completo (padre) para proveer mejor contexto."),
        ("Re-ranking:", "Añadir un modelo Cross-Encoder (ej. BGE-Reranker o Cohere Rerank) entre el Retriever y el Generator. ChromaDB puede retornar 20 resultados, y el Re-ranker los re-evalúa y filtra los 5 mejores. Esto mejora dramáticamente la precisión."),
        ("Embeddings:", "El modelo 'all-MiniLM-L6-v2' es excelente para prototipos, pero en producción en español se recomiendan modelos multilingües más potentes, como 'multilingual-e5-large' o usar APIs de embeddings dedicadas (OpenAI, Cohere).")
    ]
    for item, desc in rag_quality:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(item).bold = True
        p.add_run(f" {desc}")

    doc.add_heading('4.3 Seguridad y Fiabilidad', level=2)
    security = [
        ("Autenticación y Autorización:", "Proteger la API de FastAPI con JWT o OAuth2. Implementar 'Rate Limiting' (Limitación de Tasa) usando Redis para prevenir abusos o ataques DoS en los endpoints de inferencia LLM."),
        ("Control de Acceso a Documentos (RBAC):", "Filtrado de metadatos en ChromaDB para asegurar que un usuario solo pueda buscar respuestas en los documentos a los que tiene permisos de lectura."),
        ("Validación y Sanitización:", "Sanitizar exhaustivamente los inputs de texto para prevenir 'Prompt Injection' o ataques al LLM.")
    ]
    for item, desc in security:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(item).bold = True
        p.add_run(f" {desc}")

    doc.add_heading('4.4 Observabilidad y Mantenimiento', level=2)
    obs = [
        ("Trazabilidad RAG:", "Integrar herramientas especializadas de observabilidad para LLMs como Langfuse, Arize Phoenix o LangSmith. Esto permite ver exactamente qué contexto fue recuperado para cada prompt y medir latencias por paso."),
        ("Monitoreo de Infraestructura:", "Instrumentar la API de FastAPI con Prometheus y graficar en Grafana (Métricas de latencia de Ollama, tiempos de ChromaDB, uso de CPU/GPU)."),
        ("Evaluación Continua:", "Implementar frameworks como RAGAS (RAG Assessment) o TruLens para medir automáticamente métricas clave como: Fidelidad (Faithfulness), Relevancia de la Respuesta y Relevancia del Contexto.")
    ]
    for item, desc in obs:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(item).bold = True
        p.add_run(f" {desc}")

    doc.add_heading('5. Plan de Despliegue', level=1)
    doc.add_paragraph("Se sugiere dockerizar completamente la aplicación (Dockerfile para la API web y el generador, otro para el motor de bases de datos vectoriales). Usar Docker Compose para desarrollo y Kubernetes (Helm) o servicios Serverless de contenedores para producción. Configurar pipelines de CI/CD (GitHub Actions / GitLab CI) para ejecutar tests unitarios sobre el chunker y retriever antes de cada despliegue.")

    # Guardar el documento
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Documentacion_RAG_Produccion.docx')
    doc.save(output_path)
    print(f"Documento generado exitosamente en: {output_path}")

if __name__ == '__main__':
    main()
