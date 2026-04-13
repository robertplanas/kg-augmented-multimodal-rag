# Plan: Complete `populate_KG.py` with all preprocesses from `dev3.ipynb`

## Goal
Port the full preprocessing pipeline currently done in `dev3.ipynb` into production code invoked by `populate_KG.py`, with reusable logic allocated by category:
- `utils/knowledge_graph.py`: KG extraction + graph-source integration utilities
- `utils/node_standarization.py`: node normalization, deduplication, canonicalization, and relationship rewrite

## Current Gaps Identified
1. `populate_KG.py` only runs extraction and translation; it does not run the remaining `dev3.ipynb` preprocesses.
2. `populate_KG.py` calls `translate_nodes(...)` but does not import it.
3. No reusable module implementation exists for the `dev3.ipynb` steps:
- lemmatization integration in the runtime pipeline
- node embedding generation for all extracted head/tail nodes
- similarity-based grouping (union-find)
- representative node creation and mapping back to relationships
- optional reuse of existing Neo4j nodes to keep names/descriptions stable
4. No pipeline-level artifact persistence (`graph_raw_data.pkl`, translated/lemmatized/clean outputs) in the script.

## Scope to Implement
### A) `utils/node_standarization.py` (Normalization category)
Add reusable data structures and functions for node preprocessing:

1. **Data classes / models**
- `ExtractedNode` (or equivalent): canonical in-memory node object used during preprocessing.
  - fields: `name`, `type`, `description`, `node_id`, `embedding`, `group_id`
- `RepresentativeNodeResult` (or equivalent): representative output per group.
  - fields: `group_id`, `name`, `type`, `description`, `mapping_node_ids`

2. **Embedding and grouping utilities**
- `build_nodes_from_relationships(document_relationships, embedding_model_name, embedding_provider, ...)`
  - creates head/tail nodes from relationships
  - assigns deterministic temporary relation-local IDs (`head_id`, `tail_id`) when missing
  - computes embeddings for each extracted node
- `UnionFind` utility for grouping
- `group_nodes_by_similarity(nodes, threshold=0.05)`
  - pairwise cosine distance grouping
  - assigns `group_id`

3. **Representative selection utilities**
- `aggregate_group_metadata(group_nodes, description_llm, ...)`
  - majority vote for name/type
  - description consolidation (single description passthrough, multi-description LLM summary)
- `match_group_to_existing_nodes(group_nodes, existing_nodes, threshold=0.05)`
  - if close to existing node, reuse existing node identity
- `build_representatives(nodes_by_group, existing_nodes, ...)`

4. **Relationship rewrite utility**
- `apply_representatives_to_relationships(document_relationships, representatives)`
  - replace `head/head_type/head_description` and `tail/...` based on `head_id`/`tail_id`

5. **Top-level standardization pipeline**
- `standardize_and_canonicalize_nodes(...)`
  - sequentially runs:
    1. translate nodes
    2. lemmatize nodes/types
    3. build embeddings + group similar nodes
    4. resolve representatives (optionally against existing nodes)
    5. rewrite relationships with representative values
  - returns cleaned `document_relationships`

### B) `utils/knowledge_graph.py` (KG category)
Add graph-source/extraction-facing utilities needed by the preprocessing pipeline:

1. **Neo4j existing-node fetch helper**
- `fetch_existing_neo4j_nodes(uri, auth, database="neo4j")`
  - returns lightweight node records (`name`, `type`, `description`)
  - resilient to non-entity nodes (skip missing required properties)

2. **Optional adapter for preprocessing input**
- `relationships_to_minimal_node_records(...)` only if needed for clean boundary between extraction output and standardization input.

Rationale for placement:
- Neo4j I/O belongs to graph integration (`knowledge_graph.py`)
- translation/lemmatization/embedding dedup/canonicalization belongs to node normalization (`node_standarization.py`)

## `populate_KG.py` Refactor Plan
1. **CLI extensions**
Add args for preprocess/runtime controls:
- translation: provider/model/workers/batch size (optional overrides)
- embeddings for dedup: provider/model
- representative description model/provider
- similarity thresholds (`group_threshold`, `existing_node_threshold`)
- Neo4j connection (`--neo4j_uri`, `--neo4j_user`, `--neo4j_password`, `--neo4j_database`, `--reuse_existing_nodes`)
- artifact output path (`--output_file`) and optional intermediate saves

2. **Pipeline orchestration (in `_run_kg_pipeline`)**
- load docs from retriever
- run `convert_to_graph_elements_pipeline(...)` -> `graph_raw_data`
- run node standardization pipeline -> `graph_clean_data`
- persist artifacts:
  - always write final clean file (default `graph_clean_data.pkl`)
  - optionally write intermediate raw/translated/lemmatized files

3. **Safety + logging**
- explicit step-level logs with counts
- fail-fast for invalid thresholds/config
- graceful behavior when Neo4j is unavailable and `reuse_existing_nodes=False`

## Compatibility Notes
1. Preserve existing `Relationship` schema from `utils/knowledge_graph.py`.
2. Keep default behavior local-first (Ollama defaults) to match existing project conventions.
3. Avoid changing downstream scripts that already consume `graph_clean_data.pkl`.

## Validation Plan
1. **Smoke test (local DB present)**
- Run `populate_KG.py` end-to-end and ensure clean pickle is produced.
2. **Determinism checks**
- Verify each relationship has stable `head`/`tail` post-canonicalization.
3. **Regression checks**
- Confirm translation + lemmatization still work as before.
4. **Neo4j optional path**
- Test with `--reuse_existing_nodes` on/off.

## Implementation Order
1. Add missing reusable preprocessing utilities in `node_standarization.py`.
2. Add Neo4j existing-node fetch utility in `knowledge_graph.py`.
3. Refactor `populate_KG.py` to orchestrate full preprocessing pipeline and persistence.
4. Run quick static checks (`python -m py_compile ...`) and one smoke run (if environment has required services).

## Deliverables
1. Updated `populate_KG.py` with full `dev3` preprocessing flow.
2. New/updated classes and helper functions in:
- `utils/node_standarization.py`
- `utils/knowledge_graph.py`
3. Plan document stored at:
- `planning/populate_KG_dev3_preprocess_plan.md`
