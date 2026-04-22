#!/bin/bash

python enrich_KG.py \
  --neo4j_uri bolt://localhost:7687 \
  --neo4j_user neo4j \
  --community_graph_name my-community-graph \
  --excluded_labels image text table file code \
  --global_gamma 0.2 \
  --mid_gamma 1.5 \
  --global_property globalCommunityId \
  --mid_property midCommunityId \
  --summary_provider openai \
  --summary_model gpt-5-nano \
  --summary_max_workers 10 \
  --embedding_provider ollama \
  --embedding_model embeddinggemma:latest \
  --neo4j_database neo4j \
  --global_communities_db ./localdb/global_communities \
  --mid_communities_db ./localdb/mid_communities \
  --node_db ./localdb/node_db 
