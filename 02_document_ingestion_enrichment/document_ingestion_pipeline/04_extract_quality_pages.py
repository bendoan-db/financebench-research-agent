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
bronze_chunk_table = chunk_extraction_config["bronze_table"]["name"]
silver_chunk_table = chunk_extraction_config["silver_table"]["name"]
silver_chunk_table_schema = chunk_extraction_config["silver_table"]["schema"]

scoring_llm_endpoint = chunk_extraction_config["silver_table"]["resolution_llm_config"]["llm_endpoint_name"]
scoring_llm_prompt = chunk_extraction_config["silver_table"]["resolution_llm_config"]["prompt"]

# COMMAND ----------

from pyspark.sql.functions import explode, concat, lit, col

bronze_chunks = (
    spark.table(f"`{catalog}`.{schema}.{bronze_chunk_table}")
    .select("*", explode("parsed_content"))
    .select("*", "col.*")
)

# COMMAND ----------

display(bronze_chunks.limit(10))

# COMMAND ----------

print(scoring_llm_prompt)

# COMMAND ----------

silver_chunks_df = bronze_chunks.selectExpr(
    "*",
    f"ai_query('{scoring_llm_endpoint}', CONCAT('{scoring_llm_prompt}', doc_content)) as quality_score"
).select(silver_chunk_table_schema)
display(silver_chunks_df.limit(20))

# COMMAND ----------

silver_chunks_df.write.mode("overwrite").saveAsTable(f"`{catalog}`.{schema}.{silver_chunk_table}")

# COMMAND ----------

display(spark.table(f"`{catalog}`.{schema}.{silver_chunk_table}"))

# COMMAND ----------

display(spark.table(f"`{catalog}`.{schema}.{silver_chunk_table}").groupBy("quality_score").count())

# COMMAND ----------

display(spark.table(f"`{catalog}`.{schema}.{silver_chunk_table}").filter(col("quality_score")==0))
