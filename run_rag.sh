#!/bin/bash

python run_rag.py \
  --data_base ./localdb \
  --node_db ./localdb/node_db \
  --mid_communities_db ./localdb/mid_communities \
  --global_communities_db ./localdb/global_communities \
  --embedding_provider ollama \
  --embedding_model embeddinggemma:latest \
  --rag_provider openai \
  --rag_model gpt-5-nano \
  --graph_summary_provider openai \
  --graph_summary_model gpt-5-nano \
  --neo4j_uri bolt://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_database neo4j \
  --top_k 5 

# Optional flags:
# --neo4j_password 123456789
# --show_context
# --query "Explain the relationship between linear model and model interpretability."
