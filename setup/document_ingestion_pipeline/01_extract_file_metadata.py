# Databricks notebook source
# MAGIC %pip install -q pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open('./ingestion_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

databricks_config = config['databricks_config']
llm_config = config['llm_config']
metadata_extraction_config = config['metadata_extraction_config']

catalog=databricks_config['catalog']
schema=databricks_config['schema']
document_volume=databricks_config['document_volume']

volume_path = f"/Volumes/{catalog}/{schema}/{document_volume}"
bronze_table=metadata_extraction_config['bronze_table']["name"]
silver_table=metadata_extraction_config['silver_table']["name"]

# COMMAND ----------

sec_documents_raw = spark.read.format("binaryFile").load(volume_path)
print("Total docs: " + str(sec_documents_raw.count()))
display(sec_documents_raw.limit(2))

# COMMAND ----------

#write docs to bronze table
sec_documents_raw.write.mode("overwrite").saveAsTable(f"`{catalog}`.{schema}.{bronze_table}")
print(f"Successfully wrote bronze docs to `{catalog}`.{schema}.{bronze_table}")

# COMMAND ----------

#extract document year and company name
sec_document_year_and_company = sec_documents_raw.selectExpr("*","ai_extract(path, array('year', 'company')) AS extracted_data")

# COMMAND ----------

#extract document type
sec_document_document_type = sec_document_year_and_company.selectExpr(
    "*",
    """
    ai_classify(path, ARRAY('10k', '8k', '10q', 'Earnings Report')) AS document_type
    """,
).select("*", "extracted_data.*").drop("extracted_data")

# COMMAND ----------

sec_document_document_type.write.mode("overwrite").saveAsTable(f"`{catalog}`.{schema}.{silver_table}")
print(f"Successfully wrote bronze docs to `{catalog}`.{schema}.{silver_table}")

# COMMAND ----------

display(spark.table(f"`{catalog}`.{schema}.{silver_table}").drop("content"))

# COMMAND ----------


