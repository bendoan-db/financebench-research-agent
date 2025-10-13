# Databricks notebook source
# MAGIC %pip install -q pyyaml

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
silver_chunk_table = chunk_extraction_config["silver_table"]["name"]
gold_chunk_table_name = chunk_extraction_config["gold_table"]["name"]
gold_chunk_table_schema = chunk_extraction_config["gold_table"]["schema"]

scoring_llm_endpoint = chunk_extraction_config["gold_table"]["summarization_llm_config"]["llm_endpoint_name"]
scoring_llm_prompt = chunk_extraction_config["gold_table"]["summarization_llm_config"]["prompt"]

# COMMAND ----------

scoring_llm_endpoint

# COMMAND ----------

from pyspark.sql.functions import col, lower
silver_doc_chunks=spark.table(f"`{catalog}`.{schema}.{silver_chunk_table}").filter(col("quality_score")=="1")

# COMMAND ----------

import uuid
def generate_uuid(text):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

generate_uuid_udf = udf(generate_uuid)

# COMMAND ----------

from pyspark.sql.functions import lit, concat, substring

gold_chunk_table = (
    silver_doc_chunks.selectExpr(
        "*",
        f"ai_query('{scoring_llm_endpoint}', CONCAT('{scoring_llm_prompt}', doc_content), modelParameters => named_struct('max_tokens', 500, 'temperature', 0.0)) as doc_summarization"
    )
    .withColumn(
        "doc_content",
        concat(
            lit("COMPANY: "),
            col("resolved_company"),
            lit("\nDOCUMENT TYPE: "),
            col("document_type"),
            lit("\nDOCUMENT YEAR: "),
            col("year"),
            lit("\nDOC SUMMARY:\n"),
            col("doc_summarization"),
            lit("\n\nDOC CONTENT:\n"),
            col("doc_content"),
        ),
    )
)

# COMMAND ----------

display(gold_chunk_table.limit(10))

# COMMAND ----------

gold_chunk_table.write.mode("overwrite").saveAsTable(f"`{catalog}`.{schema}.{gold_chunk_table_name}_temp")

# COMMAND ----------

from pyspark.sql.functions import lit, concat, substring, col, lit

temp_gold_table = spark.table(f"`{catalog}`.{schema}.{gold_chunk_table_name}_temp").withColumn(
    "chunk_id", generate_uuid_udf(substring(col("doc_summarization"), 1, 2000))
).withColumn("path", concat(col("path"), lit("-"), col("chunk_id"))).select(gold_chunk_table_schema)

# COMMAND ----------

display(temp_gold_table)

# COMMAND ----------

temp_gold_table.write.mode("overwrite").saveAsTable(f"`{catalog}`.{schema}.{gold_chunk_table_name}")

# COMMAND ----------

spark.sql(f"ALTER TABLE `{catalog}`.{schema}.{gold_chunk_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

gold_table = spark.table(f"`{catalog}`.{schema}.{gold_chunk_table_name}")
display(gold_table)

# COMMAND ----------

# MAGIC %run ./06_create_vs_index

# COMMAND ----------


