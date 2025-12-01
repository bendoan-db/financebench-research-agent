# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %pip install -q -U -r ./../../requirements.txt
# MAGIC %pip install -q pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from dbruntime.databricks_repl_context import get_context

HOSTNAME = get_context().browserHostName
USERNAME = get_context().user

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope="doan", key="db-pat-token")
os.environ['DATABRICKS_URL'] = get_context().apiUrl

# COMMAND ----------

import yaml
from databricks_langchain import ChatDatabricks

with open('./retrieval_evaluation.yaml', 'r') as file:
    config = yaml.safe_load(file)

#load global configs
databricks_config = config['databricks_config']
retriever_configs = config['retriever_configs']
agent_configs = config["agent_configs"]

llm_name = agent_configs["llm"]["endpoint_name"]

#load uc configs
catalog=databricks_config['catalog']
schema=databricks_config['schema']
mlflow_experiment=databricks_config['mlflow_experiment_name']
eval_table=databricks_config['eval_table_name']
model_name=databricks_config['model']

#load vs configs
vector_search_endpoint = retriever_configs[0]['vector_search_endpoint']
vector_search_index = retriever_configs[0]['vector_search_index']
embedding_model = retriever_configs[0]['embedding_model']

llm = ChatDatabricks(endpoint=llm_name, temperature=0.0)

# COMMAND ----------

import mlflow
from dbruntime.databricks_repl_context import get_context

experiment_fqdn = f"/Users/{get_context().user}/{mlflow_experiment}"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_fqdn)

if experiment:
    experiment_id = experiment.experiment_id
    # Create the experiment if it does not exist
else:
    experiment_id = mlflow.create_experiment(experiment_fqdn)

mlflow.set_experiment(experiment_fqdn)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate
# MAGIC [Eval Dataset in UC](https://fe-vm-vdm-classic-hkbucz.cloud.databricks.com/explore/data/vdm-classic-hkbucz_catalog/financebench/financebench_evals?o=2309167578215964&activeTab=sample)

# COMMAND ----------

eval_dataset = spark.table(f"`{catalog}`.{schema}.{eval_table}").select("inputs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation

# COMMAND ----------

# Get current notebook's workspace path and add its parent to sys.path
import sys
from pyspark.dbutils import DBUtils

dbutils = DBUtils(spark)
nb_path = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()
# nb_path example: /Users/you/project/src/my_notebook

current_dir_ws = "/Workspace" + nb_path.rsplit("/", 1)[0]  # -> /Workspace/Users/you/project/src
parent_dir_ws  = current_dir_ws.rsplit("/", 1)[0]          # -> /Workspace/Users/you/project

if parent_dir_ws not in sys.path:
    sys.path.insert(0, parent_dir_ws)

# COMMAND ----------

import mlflow
from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks
from vector_search_utils.self_querying_retriever import *

from mlflow.genai.scorers import (
    RetrievalRelevance
)

for retriever in retriever_configs:
    # Initialize the retriever tool.
    if retriever["self_querying"]:
        sq_retriever = load_self_querying_retriever(llm, databricks_config, retriever)
        vs_tool = sq_retriever.as_tool()
    else:
        vs_tool = VectorSearchRetrieverTool(
            index_name=f"{catalog}.{schema}.{retriever['vector_search_index']}",
            tool_name="docs_retriever",
            tool_description="Retrieves information about SEC filings",
            )

    def my_predict_fn(messages): # the signature corresponds to the keys in the "inputs" dict
        return vs_tool.invoke(messages[-1]["content"])

    with mlflow.start_run(run_name=retriever["retriever_id"]):
    # Run evaluation with predefined scorers
        eval_results = mlflow.genai.evaluate(
            data=eval_dataset.limit(5).toPandas(),    
            predict_fn=my_predict_fn,
            scorers=[
                RetrievalRelevance(),
            ],
        )
        mlflow.end_run()

# COMMAND ----------


