# KG-Augmented Multimodal RAG
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-red)
![Graph DB](https://img.shields.io/badge/GraphDB-Neo4j-black)
![License](https://img.shields.io/badge/License-MIT-yellow)

A local-first pipeline that combines:
- Multimodal retrieval over text, tables, images, and code.
- Knowledge graph extraction and canonicalization.
- Community discovery in Neo4j (global + mid-level).
- Graph-augmented answer generation with a CLI RAG interface.

The project processes documents (`.pdf`, `.ipynb`, `.py`), stores multimodal chunks in Chroma/LocalFileStore, writes entity relationships to Neo4j, enriches graph context through community summaries, and uses all contexts during retrieval and generation.

## Run Order
Run the pipeline in this exact order:

1. `populate_database`
2. `populate_KG`
3. `enrich_KG`
4. `run_rag`

## Architecture
1. `populate_database.py`
- Ingests documents with modality-aware parsing.
- Summarizes text/images/tables/code using an LLM.
- Persists summaries + raw payloads to a local multi-vector store.

2. `populate_KG.py`
- Reads ingested docstore payloads.
- Extracts relationships from text/tables/images/code.
- Translates + lemmatizes nodes.
- Canonicalizes entities via embedding similarity.
- Writes cleaned relationships + provenance links into Neo4j.

3. `enrich_KG.py`
- Runs Leiden community detection through Neo4j GDS.
- Builds summaries for global and mid communities.
- Populates dedicated retrievers for communities and nodes.

4. `run_rag.py`
- Retrieves from 4 sources in parallel:
  - multimodal document retriever
  - node retriever
  - mid-level community retriever
  - global community retriever
- Expands node context from Neo4j (1-2 hop neighborhood).
- Summarizes graph context and builds a multimodal prompt.
- Returns final answer in interactive or single-query mode.

## Prerequisites
- Python 3.10+
- Neo4j + GDS plugin (provided via Docker Compose)
- A model backend:
  - Local: Ollama
  - Cloud: OpenAI / Google (Gemini)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment
Create a `.env` file in the project root (or `.venv/.env`).

```env
# Required for Neo4j-based steps unless passed as CLI flags
NEO4J_PASSWORD=123456789

# Optional provider keys (required only when using those providers)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key

# Optional defaults used by scripts
SUMMARY_LLM_PROVIDER=ollama
SUMMARY_LLM_MODEL=gemma3:12b
SUMMARY_LLM_MAX_WORKERS=1

KG_EXTRACTOR_PROVIDER=ollama
KG_EXTRACTOR_MODEL=gemma3:12b
KG_MAX_WORKERS=1

RAG_LLM_PROVIDER=ollama
RAG_LLM_MODEL=gemma3:12b
GRAPH_SUMMARY_LLM_PROVIDER=ollama
GRAPH_SUMMARY_LLM_MODEL=gemma3:latest
```

## Start Neo4j

```bash
docker compose up -d
```

Or:

```bash
./launch_database.sh
```

Default compose config exposes:
- Neo4j Browser: `http://localhost:7474`
- Bolt: `bolt://localhost:7687`

## Step 1: populate_database
Build the multimodal local retriever database from files in `./documents`.

Shell helper:

```bash
./run_populate_database.sh
```

Direct command:

```bash
python populate_database.py \
  --document_folder ./documents \
  --data_base ./localdb
```

Common options:
- `--summary_provider`
- `--summary_model`
- `--summary_max_workers`

### Demo Placeholder (Step 1)
<!-- Replace with your GIF path -->
![populate_database demo](./docs/gifs/01-populate_database.gif)

## Step 2: populate_KG
Extract and canonicalize graph relationships from the ingested docstore, then push to Neo4j.

Shell helper:

```bash
./run_populate_KG.sh
```

Direct command (minimal):

```bash
python populate_KG.py \
  --data_base ./localdb \
  --neo4j_uri bolt://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_database neo4j
```

Useful options:
- KG extraction: `--kg_provider`, `--kg_model`, `--kg_max_workers`
- Translation: `--node_translate_provider`, `--node_translate_model`
- Canonicalization: `--group_threshold`, `--existing_node_threshold`
- Reuse existing graph nodes: `--reuse_existing_nodes`
- Artifacts: `--raw_output_file`, `--translated_output_file`, `--lemmatized_output_file`, `--output_file`

### Demo Placeholder (Step 2)
<!-- Replace with your GIF path -->
![populate_KG demo](./docs/gifs/02-populate_KG.gif)

## Step 3: enrich_KG
Run community detection + summarization and persist graph-derived retrievers.

Shell helper:

```bash
./run_enrich_KG.sh
```

Direct command (minimal):

```bash
python enrich_KG.py \
  --neo4j_uri bolt://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_database neo4j
```

Outputs retriever DBs (default):
- `./localdb/global_communities`
- `./localdb/mid_communities`
- `./localdb/node_db`

### Demo Placeholder (Step 3)
<!-- Replace with your GIF path -->
![enrich_KG demo](./docs/gifs/03-enrich_KG.gif)

## Step 4: run_rag
Launch the GraphRAG CLI that combines document context + graph context.

Shell helper:

```bash
./run_rag.sh
```

Direct command (interactive):

```bash
python run_rag.py \
  --data_base ./localdb \
  --node_db ./localdb/node_db \
  --mid_communities_db ./localdb/mid_communities \
  --global_communities_db ./localdb/global_communities \
  --neo4j_uri bolt://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_database neo4j
```

Single-query mode:

```bash
python run_rag.py --query "Explain the relationship between model interpretability and linear models."
```

Useful options:
- `--rag_provider`, `--rag_model`
- `--graph_summary_provider`, `--graph_summary_model`
- `--embedding_provider`, `--embedding_model`
- `--top_k`
- `--show_context`

### Demo Placeholder (Step 4)
<!-- Replace with your GIF path -->
![run_rag demo](./docs/gifs/04-run_rag.gif)

## Project Structure

```text
.
├── populate_database.py        # Step 1: ingestion + multimodal DB population
├── populate_KG.py              # Step 2: KG extraction + canonicalization + Neo4j update
├── enrich_KG.py                # Step 3: community detection + community/node retrievers
├── run_rag.py                  # Step 4: Graph-augmented multimodal RAG CLI
├── run_populate_database.sh
├── run_populate_KG.sh
├── run_enrich_KG.sh
├── run_rag.sh
├── docker-compose.yml          # Neo4j + GDS container
├── documents/                  # Input documents (.pdf, .ipynb, .py)
├── local_tokenizer/            # Local tokenizer assets for Docling chunking
├── localdb/                    # Generated retriever DBs (created at runtime)
└── utils/
    ├── ingest.py               # Modality-specific parsing and extraction
    ├── summarize.py            # Parallel summarization orchestration
    ├── objects.py              # Text/Table/Image/Code objects and summarizers
    ├── knowledge_graph.py      # KG extraction chains and pipeline
    ├── node_standarization.py  # Translation, lemmatization, canonicalization
    ├── database_utils.py       # Chroma + docstore setup and population
    ├── models.py               # Provider-agnostic LLM/embedding wrappers
    └── rag.py                  # Legacy/simple RAG helpers
```

## Notes
- `populate_KG.py`, `enrich_KG.py`, and `run_rag.py` need Neo4j credentials (`--neo4j_password` or `NEO4J_PASSWORD`).
- OpenAI/Google providers require corresponding API keys.
- If you only use Ollama, ensure your local models are available before running the pipeline.

## License
MIT. See `LICENSE`.
