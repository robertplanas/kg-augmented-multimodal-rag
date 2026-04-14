#!/bin/bash

python populate_KG.py \
  -db ./localdb \
  --kg_provider openai \
  --kg_model gpt-5-nano \
  --kg_max_workers 1 \
  --node_translate_provider openai \
  --node_translate_model gpt-5-nano \
  --node_translate_max_workers 10 \
  --node_translate_batch_size 30 \
  --node_embedding_provider openai \
  --node_embedding_model text-embedding-3-small \
  --node_description_provider openai \
  --node_description_model gpt-5-nano \
  --node_description_max_workers 10 \
  --group_threshold 0.05 \
  --existing_node_threshold 0.05 \
  --neo4j_uri bolt://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_database neo4j \
  --raw_output_file graph_raw_data.pkl \
  --translated_output_file graph_translated_data.pkl \
  --lemmatized_output_file graph_lemmatized_data.pkl \
  --output_file graph_clean_data.pkl \
  --reuse_existing_nodes
