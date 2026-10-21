from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss",
    ".json", ".md", ".txt", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".sql", ".sh", ".bash", ".c", ".h", ".cpp", ".hpp", ".java",
    ".go", ".rs", ".php", ".rb", ".vue", ".svelte", ".xml", ".gradle",
}
IGNORED_PARTS = {
    ".git", ".svn", ".hg",
    "node_modules", "bower_components",
    "dist", "build", "out", "output", ".output", "bin", "obj", "target", "vendor",
    ".next", ".nuxt", ".svelte-kit", ".angular", ".turbo",
    ".venv", "venv", "env", ".env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "coverage", ".nyc_output",
    ".idea", ".vscode", ".vs",
    ".gradle", ".cargo", ".cache", "pods",
}
IGNORED_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "composer.lock", "cargo.lock", "poetry.lock", "pipfile.lock", "gemfile.lock",
}
MAX_ZIP_BYTES = 100 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024
MAX_FILES = 5000


def is_allowed_path(path: PurePosixPath, file_size: int = 0) -> bool:
    if path.is_absolute() or ".." in path.parts:
        return False
    if any(part.lower() in IGNORED_PARTS for part in path.parts):
        return False
    name_lower = path.name.lower()
    if name_lower in IGNORED_NAMES or name_lower.endswith((".pem", ".key", ".p12", ".cert", ".crt")):
        return False
    if name_lower.endswith((".min.js", ".min.css", ".bundle.js", ".map")):
        return False
    if file_size > MAX_FILE_BYTES:
        return False
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def _is_allowed(path: PurePosixPath, file_size: int = 0) -> bool:
    return is_allowed_path(path, file_size)


def zip_to_documents(zip_bytes: bytes) -> tuple[list[Document], dict]:
    """Read source files directly from ZIP; never extract untrusted paths to disk."""
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise ValueError("ZIP is larger than the 100 MB upload limit.")

    documents: list[Document] = []
    skipped = 0
    total_seen = 0

    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("This file is not a valid ZIP archive.") from exc

    with archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) > MAX_FILES:
            raise ValueError(f"ZIP contains more than {MAX_FILES} files.")

        for member in members:
            path = PurePosixPath(member.filename)
            if not _is_allowed(path, member.file_size):
                skipped += 1
                continue
            if member.file_size == 0:
                continue

            raw = archive.read(member)
            # Skip binary/non-UTF8 files rather than corrupting their contents.
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                skipped += 1
                continue

            # Basic safeguard against accidentally indexing binary-like data.
            if "\x00" in content:
                skipped += 1
                continue

            total_seen += 1
            documents.append(Document(
                page_content=content,
                metadata={"source": str(path), "file_name": path.name},
            ))

    if not documents:
        raise ValueError(
            "No supported text/code files found. Try a ZIP containing source files "
            "(for example .py, .ts, .tsx, .js, .md, or .json)."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        add_start_index=True,
        separators=["\nclass ", "\ndef ", "\nfunction ", "\nexport ", "\n\n", "\n", " ", ""],
    )

    raw_chunks = splitter.split_documents(documents)
    chunks: list[Document] = []
    doc_lookup = {doc.metadata["source"]: doc.page_content for doc in documents}

    for chunk in raw_chunks:
        if not chunk.page_content.strip():
            continue
        start_index = int(chunk.metadata.get("start_index", 0))
        original = doc_lookup.get(chunk.metadata["source"], "")
        chunk.metadata["start_line"] = original[:start_index].count("\n") + 1
        chunk.metadata["end_line"] = (
            chunk.metadata["start_line"] + chunk.page_content.count("\n")
        )
        chunks.append(chunk)

    return chunks, {
        "files_indexed": total_seen,
        "chunks_created": len(chunks),
        "files_skipped": skipped,
    }

