# RepoRAG

A local-first RAG app for asking questions about a source-code repository ZIP, with an interactive Obsidian-style graph of local import relationships. Uses Streamlit, LangGraph, Ollama, ChromaDB and PyVis.

## Requirements

- Python 3.11 or 3.12 recommended
- Ollama installed and running
- Enough RAM for your chosen local models

## 1. Pull local models

```bash
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
ollama list
```

You can choose other Ollama models in the app sidebar. The embedding model must be an embedding model; changing it after indexing means re-indexing.

## 2. Create environment and install dependencies

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a ZIP, and ask a question.

## Current MVP limitations

- Retrieval uses vector similarity only; hybrid BM25 search is a future improvement.
- Chunks are character-based rather than fully syntax-aware.
- Citations point to source chunk line ranges, not yet clickable editor links.
- The evidence check is intentionally basic; this MVP does not have a separate LLM grader.
- Chroma data is stored in `.chroma/` locally.
- Only common text/code file extensions are indexed. Large files and common dependency/build folders are skipped.

## Safety notes

This tool reads untrusted repository content. Do not upload secrets or run code from a repository. The ZIP reader does not extract archive paths to disk and skips common secret-like filenames, but this is not a guarantee that all secrets will be detected.


## Knowledge graph

The interactive graph detects common Python imports and relative JavaScript/TypeScript imports. It is a heuristic, not a complete static analyzer; unsupported import styles may not appear.
