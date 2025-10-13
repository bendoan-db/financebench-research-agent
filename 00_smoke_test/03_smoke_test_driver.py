# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %pip install -q -U -r requirements.txt
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

with open('./smoke_test_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#load global configs
databricks_config = config['databricks_config']
retriever_config = config['retriever_config']
agent_configs = config["agent_configs"]

llm_name = agent_configs["llm"]["endpoint_name"]

#load uc configs
catalog=databricks_config['catalog']
schema=databricks_config['schema']
mlflow_experiment=databricks_config['mlflow_experiment_name']
model_name=databricks_config['model']

#load vs configs
vector_search_endpoint = retriever_config['vector_search_endpoint']
vector_search_index = retriever_config['vector_search_index']
embedding_model = retriever_config['embedding_model']

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

# MAGIC %run ./02_smoke_test_agent

# COMMAND ----------

from IPython.display import display, Image
 
display(Image(AGENT.agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

example_input = {
        "messages": [
            {
                "role": "user",
                "content": "What were some key risks highlighted for for AAPL in 2020?",
            }
        ]
    }

# COMMAND ----------

from vector_search_utils.self_querying_retriever import load_self_querying_retriever

sq_retriever = load_self_querying_retriever(llm, databricks_config, retriever_config)
sq_retriever.invoke(example_input)

# COMMAND ----------

response = AGENT.predict(example_input)
print(response.messages[0].content)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model

# COMMAND ----------

print(os.path.join(os.getcwd(), "02_smoke_test_agent"))

# COMMAND ----------

import mlflow
from mlflow.models.resources import (
  DatabricksVectorSearchIndex,
  DatabricksServingEndpoint,
  DatabricksSQLWarehouse,
  DatabricksFunction,
  DatabricksGenieSpace,
  DatabricksTable,
  DatabricksUCConnection
)

with mlflow.start_run():
    logged_chain_info = mlflow.pyfunc.log_model(
        python_model=os.path.join(os.getcwd(), "02_smoke_test_agent"),
        model_config=os.path.join(os.getcwd(), "smoke_test_config.yaml"), 
        name=model_name,  # Required by MLflow
        code_paths=[
          "/Workspace/Users/ben.doan@databricks.com/financebench-research-agent/supervisor_utils", 
          "/Workspace/Users/ben.doan@databricks.com/financebench-research-agent/vector_search_utils"
        ],
        input_example=example_input,
        resources=[
        DatabricksVectorSearchIndex(index_name=f"{catalog}.{schema}.{vector_search_index}"),
        DatabricksServingEndpoint(endpoint_name=llm_name),
        DatabricksServingEndpoint(endpoint_name=embedding_model)
        ],
        pip_requirements=["-r requirements.txt"],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_chain_info.run_id}/{model_name}",
    input_data=example_input,
    env_manager="uv",
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

from databricks import agents

agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=uc_registered_model_info.version,
    environment_vars={
        "DATABRICKS_URL": get_context().apiUrl,
        "DATABRICKS_TOKEN": dbutils.secrets.get(scope="doan", key="db-pat-token")
    },
)
