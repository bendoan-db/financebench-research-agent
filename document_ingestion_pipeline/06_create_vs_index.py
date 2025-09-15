# Databricks notebook source
# MAGIC %pip install -q databricks-vectorsearch pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open('./ingestion_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
  
#load global configs
databricks_config = config['databricks_config']
chunk_extraction_config = config['chunk_extraction_config']

#load uc configs
catalog=databricks_config['catalog']
schema=databricks_config['schema']
gold_chunk_table_name = chunk_extraction_config["gold_table"]["name"]
gold_chunk_table_schema = chunk_extraction_config["gold_table"]["schema"]

#load vector search config
gold_chunk_vs_endpoint = chunk_extraction_config["gold_table"]["vector_search_config"]["endpoint_name"]
gold_chunk_index_name = chunk_extraction_config["gold_table"]["vector_search_config"]["index_name"]
gold_chunk_embedding_model = chunk_extraction_config["gold_table"]["vector_search_config"]["embedding_model"]
gold_chunk_id_column = chunk_extraction_config["gold_table"]["vector_search_config"]["id_column"]
gold_chunk_text_column = chunk_extraction_config["gold_table"]["vector_search_config"]["text_column"]

# COMMAND ----------

gold_chunk_table = spark.table(f"`{catalog}`.{schema}.{gold_chunk_table_name}")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

index = client.create_delta_sync_index(
  endpoint_name=gold_chunk_vs_endpoint,
  source_table_name=f"`{catalog}`.{schema}.{gold_chunk_table_name}",
  index_name=f"`{catalog}`.{schema}.{gold_chunk_index_name}",
  pipeline_type="TRIGGERED",
  primary_key=gold_chunk_id_column,
  embedding_source_column=gold_chunk_text_column,
  embedding_model_endpoint_name=gold_chunk_embedding_model
)

# COMMAND ----------


