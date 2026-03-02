"""
document_loader.py — Módulo de carga de documentos para el sistema RAG.

Carga archivos de texto (.txt, .md) desde un directorio y los estructura
como diccionarios con contenido y metadatos para su posterior procesamiento.
"""

import os
from pathlib import Path
from typing import Optional


class DocumentLoader:
    """Carga documentos de texto desde un directorio del sistema de archivos."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".log", ".py", ".json"}

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: Ruta al directorio raíz que contiene los documentos.
        
        Raises:
            FileNotFoundError: Si el directorio no existe.
            NotADirectoryError: Si la ruta no es un directorio.
        """
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.base_dir}")
        if not self.base_dir.is_dir():
            raise NotADirectoryError(f"No es un directorio: {self.base_dir}")

    def load_file(self, file_path: Path) -> Optional[dict]:
        """
        Carga un único archivo y retorna su contenido con metadatos.

        Args:
            file_path: Ruta absoluta al archivo.

        Returns:
            Diccionario con 'content' y 'metadata', o None si falla la lectura.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                return None

            stat = file_path.stat()
            return {
                "content": content,
                "metadata": {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "size_bytes": stat.st_size,
                    "relative_path": str(file_path.relative_to(self.base_dir)),
                },
            }
        except (UnicodeDecodeError, PermissionError, OSError) as e:
            print(f"  [WARN] No se pudo leer '{file_path}': {e}")
            return None

    def load_directory(self, recursive: bool = True) -> list[dict]:
        """
        Carga todos los documentos soportados del directorio base.

        Args:
            recursive: Si es True, busca en subdirectorios también.

        Returns:
            Lista de diccionarios con 'content' y 'metadata' por cada documento.
        """
        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(self.base_dir.glob(pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            doc = self.load_file(file_path)
            if doc is not None:
                documents.append(doc)

        return documents

    def load_single(self, filename: str) -> Optional[dict]:
        """
        Carga un archivo específico dentro del directorio base.

        Args:
            filename: Nombre o ruta relativa del archivo dentro de base_dir.

        Returns:
            Diccionario con contenido y metadatos, o None si no se encuentra.
        """
        target = self.base_dir / filename
        if not target.exists() or not target.is_file():
            return None
        if target.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return None
        return self.load_file(target)

    def get_supported_files(self, recursive: bool = True) -> list[str]:
        """Lista los archivos soportados sin cargar su contenido."""
        pattern = "**/*" if recursive else "*"
        return [
            str(f.relative_to(self.base_dir))
            for f in sorted(self.base_dir.glob(pattern))
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
