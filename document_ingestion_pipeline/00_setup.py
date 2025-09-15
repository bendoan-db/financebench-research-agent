# Databricks notebook source
# MAGIC %pip install -q pyyaml
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
volume=databricks_config['document_volume']

# COMMAND ----------

try:
  spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.{schema}")
  print(f"successfully created schema `{catalog}`.{schema}")
except Exception as e:
  print(f"Failed to create schema: {schema} with exception {e}")

# COMMAND ----------

try:
  spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.{schema}.{volume}")
except Exception as e:
  print(f"Failed to create schema: {schema} with exception: {e}")

# COMMAND ----------

import os
docs_path = os.getcwd().replace("document_ingestion_pipeline", "data/pdf_docs")
print(docs_path)

# COMMAND ----------

from pyspark.sql.functions import input_file_name
import shutil
import os

# Get the list of files in the docs_path directory
files = [f for f in os.listdir(docs_path) if os.path.isfile(os.path.join(docs_path, f))]

# Define the destination path in the Databricks volume
destination_path = f"/Volumes/{catalog}/{schema}/{volume}/"

# Copy each file to the Databricks volume
for file in files:
    shutil.copy(os.path.join(docs_path, file), destination_path)
