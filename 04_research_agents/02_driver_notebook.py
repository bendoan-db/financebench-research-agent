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

with open('./agent_config.yaml', 'r') as file:
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
eval_table=databricks_config['eval_table_name']
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

# MAGIC %run ./01_document_research_agent

# COMMAND ----------

from IPython.display import display, Image
 
display(Image(AGENT.agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

example_input = {
        "messages": [
            {
                "role": "user",
                "content": "What was the change in operating income for AAPL between 2019 and 2020, and what caused this trend?",
            }
        ]
    }

# COMMAND ----------

response = AGENT.predict(example_input)
print(response.messages[0].content)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate
# MAGIC [Eval Dataset in UC](https://fe-vm-vdm-classic-hkbucz.cloud.databricks.com/explore/data/vdm-classic-hkbucz_catalog/financebench/financebench_evals?o=2309167578215964&activeTab=sample)

# COMMAND ----------

eval_dataset = spark.table(f"`{catalog}`.{schema}.{eval_table}")

# COMMAND ----------

eval_dataset.count()

# COMMAND ----------

display(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation

# COMMAND ----------

from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
    ChatAgentRequest
)

from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    Safety,
    Guidelines,
    RetrievalRelevance
)

def my_predict_fn(messages): # the signature corresponds to the keys in the "inputs" dict
  return AGENT.predict(
    messages=[ChatAgentMessage(**message) for message in messages]
  )

# Run evaluation with predefined scorers
eval_results = mlflow.genai.evaluate(
    data=eval_dataset.toPandas(),
    predict_fn=my_predict_fn,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Safety(),
        RetrievalRelevance(),
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model

# COMMAND ----------

print(os.path.join(os.getcwd(), "01_document_research_agent"))

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
        python_model=os.path.join(os.getcwd(), "01_document_research_agent"),
        model_config=os.path.join(os.getcwd(), "agent_config.yaml"), 
        name=model_name,  # Required by MLflow
        code_paths=[os.path.join(os.getcwd(), "vector_search_utils"), os.path.join(os.getcwd(), "supervisor_utils")],
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

# COMMAND ----------

# MAGIC %md
# MAGIC * [Model Serving Endpoint](https://fe-vm-vdm-classic-hkbucz.cloud.databricks.com/ml/endpoints/agents_vdm-classic-hkbucz_catalog-financebench-financebench_res/traces?o=2309167578215964)
