# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Purpose
# MAGIC
# MAGIC - Evaluation Agent Bricks RAG Agents

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install -q -U -r requirements.txt
# MAGIC %pip install uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Eval Dataset for `mlflow.genai.evaluate`

# COMMAND ----------

import yaml

with open('./agent_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#load global configs
databricks_config = config['databricks_config']
retriever_config = config['retriever_config']

#load uc configs
catalog=databricks_config['catalog']
schema=databricks_config['schema']
mlflow_experiment=databricks_config['mlflow_experiment_name']
eval_table=databricks_config['eval_table_name']
model_name=databricks_config['model']

# COMMAND ----------

eval_dataset = spark.table(f"`{catalog}`.{schema}.{eval_table}")
display(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tr

# COMMAND ----------

df_eval = eval_dataset.toPandas()
df_eval['inputs'] = df_eval['inputs'].apply(lambda x: {'question': x['messages'][0]['content']})
display(df_eval)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Knowledge Assistant
# MAGIC
# MAGIC * Reading the RAW PDF files
# MAGIC * Agent Bricks Endpoint: `ka-7dcc8097-endpoint`

# COMMAND ----------

mlflow_experiment="ka-financebench-evaluation"

# COMMAND ----------

from openai import OpenAI
import os

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

CLIENT = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://fe-vm-vdm-classic-hkbucz.cloud.databricks.com/serving-endpoints"
)

response = CLIENT.responses.create(
    model="ka-e5c2bcc7-endpoint",
    input=[
        {
            "role": "user",
            "content": "What was AAPL's operating income in 2022?"
        }
    ]
)

print(response.output[0].content[0].text)

# COMMAND ----------

structure = """The response must use clear, concise language and structures responses logically. It avoids jargon or explains technical terms when used."""

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
import mlflow

os.environ["RAG_EVAL_MAX_WORKERS"] = "1" 

def my_predict_fn(question):
    response = CLIENT.responses.create(
        model="ka-e5c2bcc7-endpoint",
        input=[
            {
                "role": "user",
                "content": question
            }
        ]
    )
    return response.output[0].content[0].text

# Run evaluation with predefined scorers
eval_results = mlflow.genai.evaluate(
    data=df_eval,
    predict_fn=my_predict_fn,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Safety(),
        RetrievalRelevance()
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC Getting 429 rate limit exceeded error

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run in sequence

# COMMAND ----------

eval_results = []

for _, row in df_eval.iterrows():
    result = mlflow.genai.evaluate(
        data=row.to_frame().T,
        predict_fn=my_predict_fn,
        scorers=[
            Correctness(),
            RelevanceToQuery(),
            Safety(),
            figure_correctness,
            Guidelines(name="structure", guidelines=structure),
        ],
    )
    eval_results.append(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## multi-agent supervisor
# MAGIC
# MAGIC * Agent Bricks Endpoint: `mas-604fc27d-endpoint`

# COMMAND ----------

mlflow.set_experiment('/Users/q.yu@databricks.com/mlflow_experiments/agent_bricks_ma_eval')


def my_predict_fn(question):
    response = CLIENT.responses.create(
        model="mas-604fc27d-endpoint",
        input=[
            {
                "role": "user",
                "content": question
            }
        ]
    )
    return response.output[0].content[0].text

eval_results = []
for _, row in df_eval.iterrows():
    result = mlflow.genai.evaluate(
        data=row.to_frame().T,
        predict_fn=my_predict_fn,
        scorers=[
            Correctness(),
            RelevanceToQuery(),
            Safety(),
            figure_correctness,
            Guidelines(name="structure", guidelines=structure),
        ],
    )
    eval_results.append(result)

# COMMAND ----------


