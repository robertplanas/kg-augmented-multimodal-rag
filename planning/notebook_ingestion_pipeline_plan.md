# Notebook Ingestion Plan (PDF-like: chunks + tables + images + context)

## Goal
Extend the ingestion pipeline so `.ipynb` files produce the same output contract as PDFs:
- `text_objs`
- `table_objs`
- `images_objs`

The notebook pipeline must preserve contextual windows (similar to `find_context_for_element` in `objects.py`) so image/table summarization gets nearby notebook context.

## Current Gaps To Fix
1. `ingest_ipynb_document` is incomplete/incompatible:
- Uses `NotebookLoader` and then `HybridChunker` on a LangChain `Document`, which is not a Docling document.
- Returns only `chunks`, not `(text_objs, table_objs, images_objs)`.
2. Notebook support is not aligned with downstream contract (`summarize_objects`, `populate_database`).
3. `.py` routing bug in `ingest_document`:
- `if document_path.endswith(".py")` incorrectly calls `ingest_ipynb_document` instead of `ingest_py_document`.
4. `TextChunk` currently assumes Docling metadata shape and will lose notebook metadata unless extended.

## Design Principles
1. Keep notebook ingestion output schema consistent with PDF ingestion.
2. Reuse existing summarization flow (`summarize_objects`) without changing its API.
3. Preserve ordering across notebook elements so context windows are deterministic.
4. Avoid fragile parsing; add optional richer parsing only if needed.

## Target Architecture

### 1. Build a Linear Notebook Element Stream
Create an internal normalized element list from `nbformat`:
- `text` elements:
  - markdown cell source
  - code cell source
  - text outputs (`stream`, `text/plain`, tracebacks)
- `table` elements:
  - markdown tables in markdown cells
  - output tables from `text/html`, `text/markdown`, `text/csv`, `application/json` (tabular-like payloads)
- `image` elements:
  - output MIME types `image/png`, `image/jpeg`, `image/webp`, `image/gif`
  - markdown attachments (`cell["attachments"]`) with image MIME

Each normalized element should include:
- `kind`: `text|table|image`
- `content`: text markdown/base64
- `cell_index`
- `output_index` (if output-originated)
- `cell_type`
- `source_filename`
- `mime_type` (for images / output-derived tables)
- `sequence_index` (global position in notebook stream)

This `sequence_index` is the anchor for context-window retrieval.

### 2. Notebook Context Window (PDF-like behavior)
Implement context builder equivalent to `DocumentObject.find_context_for_element` but for normalized notebook elements.

Suggested helper:
- `build_notebook_context(elements, target_sequence_index, tokenizer, max_context_len=512) -> str`

Algorithm:
1. Start with target content tokens.
2. If target exceeds `max_context_len`, center-crop tokens (reuse `get_central_tokens`).
3. Use remaining budget for previous and next elements, split roughly half/half.
4. Return `prev + target + next`.

This gives image/table objects local semantic context similar to PDF item context.

### 3. Notebook Objects in `objects.py`
Add notebook-native object classes that match downstream expectations:

1. `NotebookTableObject`
- Fields: `markdown`, `context`, `metadata`, `description`
- Method: `summarize_table(...)` (same logic as `TableObject`)

2. `NotebookImageObject`
- Fields: `base64`, `context`, `metadata`, `description`
- Method: `summarize_image(...)` (same logic as `ImageObject`)

3. `NotebookTextChunk` (or extend existing `TextChunk` to support dict metadata)
- Fields: `text`, `metadata`, `description`
- Method: `summarize_text(...)`

Metadata format should be flat/JSON-safe (compatible with `populate_database.flatten_metadata` pattern):
- `filename`
- `cell_index`
- `output_index` (nullable)
- `cell_type`
- `mime_type` (nullable)
- `pages`: `[]` for notebooks (or omit)
- `origin`: `"ipynb"`

### 4. Chunking Strategy for Notebook Text
Use tokenizer-aware splitting, but without Docling chunker:
- Build text blocks from normalized `text` elements.
- Split using `RecursiveCharacterTextSplitter` (or token-aware splitter if needed).
- Carry cell/output metadata into each chunk.

Recommended defaults:
- `chunk_size=1000`
- `chunk_overlap=150`

This keeps behavior consistent with current notebook splitter direction while avoiding Docling dependency mismatch.

### 5. Table Extraction Strategy
Implement layered extraction in `ingest.py`:
1. Markdown cell tables:
- Detect pipe-table patterns (`| ... |` + separator row).
2. Code output tables:
- `text/markdown` or `text/csv`: normalize directly to markdown.
- `text/html`: store HTML as markdown-compatible wrapper or convert when parser available.
- `application/json`: if list-of-dicts or dict-of-lists, render to markdown table.
3. Fallback:
- If tabular heuristic fails, keep as text element to avoid data loss.

### 6. Image Extraction Strategy
For each code cell output and markdown attachment:
1. Collect base64 payload for image MIME.
2. Normalize to base64 string.
3. Deduplicate with existing `filter_images` (hash-based).
4. Build `NotebookImageObject` with context and metadata.

### 7. `ingest.py` Integration Changes
Update key functions:

1. `ingest_document`
- `.pdf` -> `ingest_pdf_document`
- `.ipynb` -> `ingest_ipynb_document`
- `.py` -> `ingest_py_document` (fix current bug)

2. `ingest_ipynb_document`
- Read notebook via `nbformat.read(...)`.
- Build normalized element stream.
- Build `text_objs`, `table_objs`, `images_objs`.
- Deduplicate notebook images.
- Return tuple `(text_objs, table_objs, images_objs)`.

3. Remove unused/incorrect pieces:
- `NotebookLoader` path for chunking with `HybridChunker`.
- Unused imports tied to old notebook flow.

### 8. `summarize.py` Impact
No API change required if notebook objects implement:
- `summarize_text`
- `summarize_image`
- `summarize_table`

The function can remain unchanged.

### 9. Ingestion Entry Script Changes (`populate_database.py`)
Current script only globs PDFs. To include notebooks in pipeline:
- Replace `glob(f"{folder}/*.pdf")` with multi-extension discovery:
  - `.pdf`
  - `.ipynb`
  - optional `.py` (if desired in same run)

This makes notebook ingestion actually reachable in batch mode.

## File-by-File Implementation Plan

1. `utils/objects.py`
- Add notebook object classes (`NotebookTableObject`, `NotebookImageObject`, `NotebookTextChunk`) or extend `TextChunk` for dict metadata.
- Add reusable context helper for ordered notebook elements.
- Keep summarization methods aligned with existing prompts.

2. `utils/ingest.py`
- Add notebook parsing + normalization helpers.
- Implement robust `ingest_ipynb_document` returning `(texts, tables, images)`.
- Fix `.py` dispatch.
- Keep image dedup shared with PDF flow.

3. `populate_database.py`
- Expand file discovery to include notebooks.

4. `utils/summarize.py`
- No logic changes expected (verify compatibility only).

## Testing Plan

### Unit Tests (new `tests/test_notebook_ingest.py`)
1. `test_ingest_ipynb_returns_triplet`
- Asserts tuple length 3 and object lists.
2. `test_notebook_extracts_text_chunks`
- Notebook with markdown + code + output text.
3. `test_notebook_extracts_images`
- Notebook with base64 PNG output.
4. `test_notebook_extracts_tables`
- Markdown table and output tabular format.
5. `test_notebook_context_window_includes_neighbors`
- Confirms context contains surrounding cells.
6. `test_notebook_image_deduplication`
- Same image repeated across outputs yields one stored image object.

### Integration Tests
1. Run full ingest + summarize on sample notebook.
2. Persist to local DB; confirm stored records for `text`, `table`, `image`.
3. Run retrieval query and ensure notebook-derived content is returned.

## Acceptance Criteria
1. `ingest_document("*.ipynb")` returns `(text_objs, table_objs, images_objs)` without exceptions.
2. Notebook tables and images have non-empty `context` derived from nearby notebook elements.
3. `summarize_objects` runs unchanged on notebook outputs.
4. `populate_database.py` ingests notebooks when present in target folder.
5. Existing PDF ingestion behavior remains unchanged.

## Risks And Mitigations
1. Notebook output variety is high.
- Mitigation: layered parsing + graceful fallback to text.
2. HTML table conversion quality may vary.
- Mitigation: preserve raw HTML/table text if conversion fails.
3. Large notebooks can create oversized contexts.
- Mitigation: strict token budget and center-crop.

## Suggested Implementation Order
1. Add notebook objects + context builder in `objects.py`.
2. Rebuild `ingest_ipynb_document` with normalized stream in `ingest.py`.
3. Fix dispatch bug for `.py`.
4. Update `populate_database.py` file discovery.
5. Add tests and run end-to-end smoke test.
