from __future__ import annotations
import io
import re
import zipfile
from pathlib import PurePosixPath
import networkx as nx
from pyvis.network import Network
from repo_rag.ingest import MAX_FILES, MAX_ZIP_BYTES, is_allowed_path

PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)
JS_IMPORT_RE = re.compile(r'''(?:import\s+(?:[^'\"]*?\s+from\s+)?|export\s+[^'\"]*?\s+from\s+|require\s*\(|import\s*\()\s*['\"]([^'\"]+)['\"]''')

def _read_zip(zip_bytes: bytes) -> dict[str, str]:
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise ValueError("ZIP is larger than the 100 MB upload limit.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("This file is not a valid ZIP archive.") from exc
    files = {}
    with archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) > MAX_FILES:
            raise ValueError(f"ZIP contains more than {MAX_FILES} files.")
        for member in members:
            path = PurePosixPath(member.filename)
            if not is_allowed_path(path, member.file_size):
                continue
            try:
                content = archive.read(member).decode("utf-8")
            except (UnicodeDecodeError, OSError, RuntimeError):
                continue
            if "\x00" not in content:
                files[str(path)] = content
    return files

def _resolve(source: str, ref: str, known: set[str]) -> str | None:
    if ref.startswith("."):
        base = str(PurePosixPath(source).parent / ref)
    else:
        base = ref.replace(".", "/")
    candidates = [base]
    if not PurePosixPath(base).suffix:
        for ext in (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte"):
            candidates.append(base + ext)
        for ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            candidates.append(base.rstrip("/") + "/index" + ext)
        candidates.append(base.rstrip("/") + "/__init__.py")
    for candidate in candidates:
        candidate = str(PurePosixPath(candidate))
        if candidate in known:
            return candidate
    return None

def build_repository_graph(zip_bytes: bytes) -> tuple[str, dict]:
    files = _read_zip(zip_bytes)
    known = set(files)
    graph = nx.DiGraph()
    graph.add_nodes_from(files)
    for path, content in files.items():
        refs = []
        suffix = PurePosixPath(path).suffix.lower()
        if suffix == ".py":
            for match in PY_IMPORT_RE.finditer(content):
                target = _resolve(path, match.group(1) or match.group(2), known)
                if target and target != path:
                    refs.append(target)
        if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte"}:
            for match in JS_IMPORT_RE.finditer(content):
                ref = match.group(1)
                if ref.startswith("."):
                    target = _resolve(path, ref, known)
                    if target and target != path:
                        refs.append(target)
        for target in set(refs):
            graph.add_edge(path, target, title="imports")
    net = Network(height="680px", width="100%", bgcolor="#111111", font_color="#eeeeee", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-25000, central_gravity=0.25, spring_length=160, spring_strength=0.025, damping=0.09)
    for node in graph.nodes:
        name = PurePosixPath(node).name
        net.add_node(node, label=name, title=f"<b>{node}</b><br>Imports: {graph.out_degree(node)}<br>Imported by: {graph.in_degree(node)}", shape="dot", size=12 + min(18, graph.degree(node) * 3), color="#e05252" if PurePosixPath(node).suffix.lower() in {".py", ".ts", ".tsx", ".js", ".jsx"} else "#6b8afd")
    for source, target in graph.edges:
        net.add_edge(source, target, title="imports", arrows="to", color="#666666")
    return net.generate_html(), {"files": graph.number_of_nodes(), "connections": graph.number_of_edges(), "isolated": sum(1 for node in graph.nodes if graph.degree(node) == 0)}
