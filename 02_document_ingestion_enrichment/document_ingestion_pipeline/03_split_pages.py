# Databricks notebook source
# MAGIC %pip install pdfplumber pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open('./ingestion_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
  
#load global configs
databricks_config = config['databricks_config']
chunk_extraction_config = config['chunk_extraction_config']
metadata_extraction_config = config['metadata_extraction_config']

#load uc configs
catalog=databricks_config['catalog']
schema=databricks_config['schema']
metadata_gold_table = metadata_extraction_config["gold_table"]["name"]

#load configs for table with chunks
bronze_chunk_table = chunk_extraction_config["bronze_table"]["name"]
bronze_chunk_table_schema = chunk_extraction_config["bronze_table"]["schema"]

# COMMAND ----------

import os
import json
from openai import OpenAI
from dbruntime.databricks_repl_context import get_context

from pyspark.sql.functions import explode, col, regexp_extract
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

# COMMAND ----------

# parse pdf using pypdf
@udf(
    ArrayType(
        StructType(
            [
                StructField("doc_content", StringType()),
                StructField("page_number", IntegerType()),
            ]
        )
    )
)
def parse_pdf(file_path):
    import pdfplumber

    try:
        final_output = []
        page_counter = 1
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                parsed_content = page.extract_text()
                page_data = {
                    "doc_content": (str(parsed_content) if parsed_content else "No Content"),
                    "page_number": page_counter,
                }
                final_output.append(page_data)
                page_counter += 1
        return final_output
    except Exception as e:
        print(f"Exception {e} has been thrown during parsing")
        return [e]

# COMMAND ----------

from pyspark.sql.functions import explode, col
from pyspark.sql.functions import regexp_extract

sec_docs = spark.table(f"`{catalog}`.{schema}.{metadata_gold_table}")

sec_docs_processed = (sec_docs
    .withColumn("cleaned_path", regexp_extract(col("path"), "dbfs:(.*)", 1))
    .withColumn("filename", regexp_extract(col("path"), "/pdf/([^/]+)$", 1))
    .withColumn("parsed_content", parse_pdf("cleaned_path"))
).select(bronze_chunk_table_schema)

# COMMAND ----------

#display(sec_docs_processed)

# COMMAND ----------

sec_docs_processed.write.mode("overwrite").saveAsTable(f"`{catalog}`.{schema}.{bronze_chunk_table}")

# COMMAND ----------

display(spark.table(f"`{catalog}`.{schema}.{bronze_chunk_table}"))
