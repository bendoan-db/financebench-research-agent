# Databricks notebook source
# MAGIC %pip install -q databricks-vectorsearch pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open('./ingestion_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#load global configs
databricks_config = config['databricks_config']
entity_resolution_config = config['entity_resolution_config']

#load uc configs
catalog=databricks_config['catalog']
schema=databricks_config['schema']
entity_table = entity_resolution_config['entity_table']

#load vs configs
vector_search_endpoint = entity_resolution_config['vector_search_endpoint']
vector_search_index = entity_resolution_config['vector_search_index']
embedding_model = entity_resolution_config['embedding_model']
vector_search_id_column = entity_resolution_config['vector_search_id_column']
embedding_source_column = entity_resolution_config['embedding_source_column']

# COMMAND ----------

import os

#load csv with company names
file_path = (os.getcwd().replace("document_ingestion_pipeline", "data/sec_companies.csv"))
company_names = spark.read.format("csv").option("header", "true").load("file://" + file_path)
display(company_names)

# COMMAND ----------

#write entities to table
company_names.write.mode("overwrite").saveAsTable(f"`{catalog}`.{schema}.{entity_table}")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

#create vector search index
spark.sql(f"ALTER TABLE `{catalog}`.{schema}.{entity_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

client = VectorSearchClient()
try:
  index = client.create_delta_sync_index(
    endpoint_name=vector_search_endpoint,
    source_table_name=f"`{catalog}`.{schema}.{entity_table}",
    index_name=f"`{catalog}`.{schema}.{vector_search_index}",
    pipeline_type="TRIGGERED",
    primary_key=vector_search_id_column,
    embedding_source_column=embedding_source_column,
    embedding_model_endpoint_name=embedding_model
  )
except Exception as e:
  print(e)
