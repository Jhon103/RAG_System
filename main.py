"""
main.py — Pipeline principal del sistema RAG.

CLI interactivo con Rich para indexar documentos, hacer consultas semánticas,
y ver estadísticas del sistema. Punto de entrada del proyecto.
"""

import sys
import os
import time

# Asegurar que los módulos locales se encuentren
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from document_loader import DocumentLoader
from chunker import TextChunker
from retriever import Retriever
from generator import ResponseGenerator
from vector_store import VectorStore


# ─────────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNK_STRATEGY = "sentence"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.15

console = Console()


def print_banner():
    """Imprime el banner principal del sistema."""
    banner_text = """
[bold cyan]██████╗  █████╗  ██████╗ [/bold cyan]
[bold cyan]██╔══██╗██╔══██╗██╔════╝ [/bold cyan]
[bold cyan]██████╔╝███████║██║  ███╗[/bold cyan]
[bold cyan]██╔══██╗██╔══██║██║   ██║[/bold cyan]
[bold cyan]██║  ██║██║  ██║╚██████╔╝[/bold cyan]
[bold cyan]╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ [/bold cyan]

[dim]Retrieval-Augmented Generation System[/dim]
[dim]Embeddings: all-MiniLM-L6-v2 | Vector Store: ChromaDB[/dim]
"""
    console.print(Panel(banner_text, border_style="cyan", box=box.DOUBLE))


def print_menu():
    """Muestra el menú principal."""
    table = Table(
        title="[bold yellow]Menú Principal[/bold yellow]",
        box=box.ROUNDED,
        border_style="yellow",
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Opción", style="bold green", width=6)
    table.add_column("Descripción", style="white")

    table.add_row("1", "📥  Indexar documentos desde ./docs/")
    table.add_row("2", "🔍  Hacer una consulta al sistema RAG")
    table.add_row("3", "📊  Ver estadísticas de la colección")
    table.add_row("4", "🗑️   Limpiar la base de datos vectorial")
    table.add_row("5", "📂  Listar documentos disponibles")
    table.add_row("0", "🚪  Salir")

    console.print(table)


def initialize_components() -> tuple[VectorStore, Retriever, ResponseGenerator]:
    """Inicializa los componentes principales del sistema."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Inicializando vector store...", total=None)
        vs = VectorStore(
            persist_directory=CHROMA_DIR,
            collection_name="rag_documents",
        )

        progress.update(task, description="[cyan]Configurando retriever...")
        ret = Retriever(
            vector_store=vs,
            top_k=TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        progress.update(task, description="[cyan]Preparando generador...")
        gen = ResponseGenerator(max_context_chunks=TOP_K)

        progress.update(task, description="[green]✓ Sistema listo")
        time.sleep(0.3)

    return vs, ret, gen


def action_index_documents(vs: VectorStore):
    """Indexa todos los documentos del directorio ./docs/"""
    console.print(f"\n[bold cyan]📥 Indexando documentos desde:[/bold cyan] {DOCS_DIR}\n")

    if not os.path.exists(DOCS_DIR):
        console.print("[red]✗ El directorio ./docs/ no existe.[/red]")
        console.print("[dim]  Crea el directorio y añade archivos .txt o .md[/dim]")
        return

    # Cargar documentos
    loader = DocumentLoader(DOCS_DIR)
    files = loader.get_supported_files()

    if not files:
        console.print("[yellow]⚠ No se encontraron documentos soportados en ./docs/[/yellow]")
        return

    console.print(f"[green]  Encontrados {len(files)} archivos:[/green]")
    for f in files:
        console.print(f"    [dim]• {f}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Fase 1: Cargar
        task = progress.add_task("[cyan]Cargando documentos...", total=None)
        documents = loader.load_directory(recursive=True)
        progress.update(task, description=f"[green]✓ {len(documents)} documentos cargados")
        time.sleep(0.2)

        # Fase 2: Chunkear
        progress.update(task, description="[cyan]Fragmentando documentos...")
        chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            strategy=CHUNK_STRATEGY,
        )
        chunks = chunker.chunk_documents(documents)
        progress.update(task, description=f"[green]✓ {len(chunks)} chunks generados")
        time.sleep(0.2)

        # Fase 3: Indexar en vector store
        progress.update(task, description="[cyan]Generando embeddings e indexando...")
        added = vs.add_documents(chunks)
        progress.update(task, description=f"[green]✓ {added} chunks indexados en ChromaDB")
        time.sleep(0.3)

    # Resumen
    console.print()
    summary_table = Table(box=box.SIMPLE, border_style="green")
    summary_table.add_column("Métrica", style="bold")
    summary_table.add_column("Valor", style="cyan")
    summary_table.add_row("Documentos cargados", str(len(documents)))
    summary_table.add_row("Chunks generados", str(len(chunks)))
    summary_table.add_row("Chunks indexados", str(added))
    summary_table.add_row("Tamaño de chunk", f"{CHUNK_SIZE} chars")
    summary_table.add_row("Overlap", f"{CHUNK_OVERLAP} chars")
    summary_table.add_row("Estrategia", CHUNK_STRATEGY)
    console.print(summary_table)


def action_query(retriever: Retriever, generator: ResponseGenerator):
    """Ejecuta una consulta al sistema RAG."""
    console.print("\n[bold cyan]🔍 Consulta al Sistema RAG[/bold cyan]")
    console.print("[dim]  Escribe tu pregunta en lenguaje natural.[/dim]\n")

    query = Prompt.ask("[bold yellow]Pregunta")
    if not query.strip():
        console.print("[red]  ✗ Pregunta vacía.[/red]")
        return

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Buscando chunks relevantes...", total=None)
        context = retriever.retrieve_with_context(query)

        progress.update(task, description="[cyan]Generando respuesta...")
        response = generator.generate(
            query=query,
            retrieval_results=context["results"],
            include_sources=True,
            include_scores=True,
        )

        progress.update(task, description="[green]✓ Respuesta generada")
        time.sleep(0.2)

    # Mostrar respuesta
    console.print()
    console.print(Panel(
        response["answer"],
        title="[bold green]Respuesta RAG[/bold green]",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2),
    ))

    # Meta-información
    meta_table = Table(box=box.SIMPLE, border_style="dim")
    meta_table.add_column("", style="dim")
    meta_table.add_column("", style="dim")
    meta_table.add_row("Chunks usados", str(response["chunks_used"]))
    meta_table.add_row("Confianza", f"{response['confidence']:.0%}")
    meta_table.add_row("Similitud promedio", f"{context['avg_similarity']:.2%}")
    meta_table.add_row("Fuentes únicas", str(len(context["sources"])))
    console.print(meta_table)


def action_stats(vs: VectorStore):
    """Muestra estadísticas de la colección."""
    console.print("\n[bold cyan]📊 Estadísticas del Vector Store[/bold cyan]\n")

    stats = vs.get_collection_stats()

    table = Table(
        title="[bold]Colección ChromaDB[/bold]",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Propiedad", style="bold yellow")
    table.add_column("Valor", style="white")

    table.add_row("Nombre de colección", stats["collection_name"])
    table.add_row("Total de chunks", str(stats["total_chunks"]))
    table.add_row("Modelo de embeddings", stats["embedding_model"])
    table.add_row("Directorio de datos", stats["persist_directory"])
    table.add_row("Tamaño de chunk config.", f"{CHUNK_SIZE} chars")
    table.add_row("Overlap config.", f"{CHUNK_OVERLAP} chars")
    table.add_row("Top-K config.", str(TOP_K))
    table.add_row("Umbral similitud", str(SIMILARITY_THRESHOLD))

    console.print(table)


def action_clear(vs: VectorStore):
    """Limpia la base de datos vectorial."""
    console.print("\n[bold red]🗑️  Limpiar Base de Datos[/bold red]\n")

    confirm = Prompt.ask(
        "[yellow]¿Estás seguro? Esto eliminará todos los datos indexados[/yellow]",
        choices=["s", "n"],
        default="n",
    )

    if confirm == "s":
        vs.delete_collection()
        console.print("[green]✓ Base de datos limpiada exitosamente.[/green]")
    else:
        console.print("[dim]  Operación cancelada.[/dim]")


def action_list_files():
    """Lista los documentos disponibles en ./docs/"""
    console.print("\n[bold cyan]📂 Documentos Disponibles[/bold cyan]\n")

    if not os.path.exists(DOCS_DIR):
        console.print("[red]✗ El directorio ./docs/ no existe.[/red]")
        return

    loader = DocumentLoader(DOCS_DIR)
    files = loader.get_supported_files()

    if not files:
        console.print("[yellow]⚠ No se encontraron documentos.[/yellow]")
        return

    table = Table(box=box.SIMPLE, border_style="cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Archivo", style="bold green")
    table.add_column("Extensión", style="yellow")

    for i, f in enumerate(files, 1):
        ext = os.path.splitext(f)[1]
        table.add_row(str(i), f, ext)

    console.print(table)
    console.print(f"\n[dim]Total: {len(files)} archivos[/dim]")


def main():
    """Punto de entrada principal del sistema RAG."""
    print_banner()

    # Inicializar componentes
    vs, retriever, generator = initialize_components()

    console.print("[green]✓ Sistema RAG inicializado correctamente.[/green]\n")

    while True:
        print_menu()
        try:
            choice = Prompt.ask("\n[bold cyan]Selecciona una opción[/bold cyan]", default="2")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Saliendo...[/dim]")
            break

        if choice == "1":
            action_index_documents(vs)
        elif choice == "2":
            action_query(retriever, generator)
        elif choice == "3":
            action_stats(vs)
        elif choice == "4":
            action_clear(vs)
        elif choice == "5":
            action_list_files()
        elif choice == "0":
            console.print("\n[bold cyan]👋 ¡Hasta luego![/bold cyan]\n")
            break
        else:
            console.print("[red]  ✗ Opción no válida.[/red]")

        console.print()


if __name__ == "__main__":
    main()
