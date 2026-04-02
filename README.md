# kg-augmented-multimodal-rag
Knowledge graph–augmented multimodal RAG pipeline for combining structured graph knowledge with text and visual retrieval to improve LLM reasoning and grounding.

## Summary LLM configuration (safe secrets)

`utils/summarize.py` can load runtime parameters from environment variables, so no API key is stored in code.

Defaults (when nothing is set):
- `SUMMARY_LLM_PROVIDER=ollama`
- `SUMMARY_LLM_MODEL=gemma3:12b`
- `SUMMARY_LLM_MAX_WORKERS=1`

Optional environment variables:
- `SUMMARY_LLM_PROVIDER`
- `SUMMARY_LLM_MODEL`
- `SUMMARY_LLM_MAX_WORKERS`
- `OPENAI_API_KEY` (used automatically when provider is `openai`)
- `GOOGLE_API_KEY` (used automatically when provider is `google` or `gemini`)

Safe usage options:
- Export in your shell profile (outside repo), e.g. `~/.zshrc`.
- Or create a local `.env` in the project root (already git-ignored).

`populate_database.py` also supports non-secret overrides:
- `--summary_provider`
- `--summary_model`
- `--summary_max_workers`

Example:
```bash
export SUMMARY_LLM_PROVIDER=openai
export SUMMARY_LLM_MODEL=gpt-4o-mini
export SUMMARY_LLM_MAX_WORKERS=4
export OPENAI_API_KEY=your_key_here
python3 populate_database.py -df ./documents
```
