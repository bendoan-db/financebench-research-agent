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
metadata_extraction_config = config['metadata_extraction_config']

#load uc configs
catalog=databricks_config['catalog']
schema=databricks_config['schema']
silver_table = metadata_extraction_config['silver_table']['name']
gold_table = metadata_extraction_config['gold_table']['name']
gold_table_columns = metadata_extraction_config['gold_table']['schema']

#load entity resolution configs
vector_search_endpoint = entity_resolution_config['vector_search_endpoint']
vector_search_index = entity_resolution_config['vector_search_index']

embedding_model = entity_resolution_config['embedding_model']
vector_search_id_column = entity_resolution_config['vector_search_id_column']
embedding_source_column = entity_resolution_config['embedding_source_column']

entity_table = entity_resolution_config['entity_table']
er_temp_table = entity_resolution_config['er_temp_table']

#entity resolution model configs
er_llm_endpoint = entity_resolution_config['resolution_llm_config']['llm_endpoint_name']
prompt = entity_resolution_config['resolution_llm_config']['prompt']

# COMMAND ----------

#load documents
silver_sec_docs = spark.table(f"`{catalog}`.{schema}.{silver_table}")
display(silver_sec_docs)

# COMMAND ----------

import pandas as pd
from typing import List, Dict
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, FloatType
from databricks.vector_search.client import VectorSearchClient

@pandas_udf(StringType())
def vector_search_lookup(texts: pd.Series) -> pd.Series:
    """
    Performs vector search lookup against a Databricks vector search index.

    This UDF:
    1. Takes input text from the company_name column
    2. Queries a vector search index to find semantically similar content
    3. Returns the top 3 matches for each input text

    Args:
        texts (pd.Series): Series of text strings to search for

    Returns:
        pd.Series: Series of arrays, each containing the top 3 matches
                  as structs with id, text, and score fields
    """
    # Initialize the vector search client
    vs_client = VectorSearchClient()

    # Configuration - in production, these would typically come from environment variables
    VS_INDEX_NAME = (
        f"{catalog}.{schema}.{vector_search_index}"  # Replace with your index name
    )
    VS_ENDPOINT_NAME = vector_search_endpoint  # Replace with your endpoint name

    # Get the vector search index
    vs_index = vs_client.get_index(
        endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME
    )

    results = []
    # Process each text in the batch
    for text in texts:
        # Handle empty/null inputs
        if text is None or pd.isna(text) or text.strip() == "":
            results.append([])
            continue

        try:
            # Query the vector search index
            search_results = vs_index.similarity_search(
                query_text=text,
                num_results=3,  # Get top 3 matches
                columns=["company_name"],
            )

            top_3_companies = "; ".join(
                [result[0] for result in search_results["result"]["data_array"]]
            )
            results.append(top_3_companies)
        except Exception as e:
            error_message = f"Error processing text: {str(e)}"
            results.append(error_message)
            #pass

    return pd.Series(results)

# COMMAND ----------

entities = spark.table(f"`{catalog}`.{schema}.{entity_table}").select("company_name").rdd.flatMap(lambda x: x).collect()
entities

# COMMAND ----------

silver_docs_with_vectors = silver_sec_docs.filter(col("company").isNotNull()).withColumn("possible_entities", vector_search_lookup(col("company")))
#display(silver_docs_with_vectors)

# COMMAND ----------

display(silver_docs_with_vectors.limit(100).withColumn("possible_entities", vector_search_lookup(col("company"))))

# COMMAND ----------

display(silver_docs_with_vectors.groupBy("company").count())

# COMMAND ----------

#we can't run ai_query and scalar udfs on the same dataframe, so we have to save it and load it back in
silver_docs_with_vectors.write.saveAsTable(f"`{catalog}`.{schema}.{er_temp_table}")

# COMMAND ----------

display(spark.table(f"`{catalog}`.{schema}.{er_temp_table}"))

# COMMAND ----------

from pyspark.sql.functions import concat, lit
silver_docs_with_entities = spark.table(f"`{catalog}`.{schema}.{er_temp_table}").withColumn(
  "resolved_entity_prompt", concat(
    lit(prompt),
    col("company"),
    lit("\nPOSSIBLE ENITITIES -> "),
    col("possible_entities"),\
    lit("\nMATCHED ENTITY ->")
  )
)
display(silver_docs_with_entities)

# COMMAND ----------

gold_docs = silver_docs_with_entities.selectExpr(
    "*", f"ai_query('{er_llm_endpoint}', resolved_entity_prompt) as resolved_company"
).select(gold_table_columns)
display(gold_docs)

# COMMAND ----------

gold_docs.write.saveAsTable(f"`{catalog}`.{schema}.{gold_table}")

# COMMAND ----------

display(spark.table(f"`{catalog}`.{schema}.{gold_table}"))

# COMMAND ----------


