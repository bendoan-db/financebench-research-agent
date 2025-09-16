# Databricks Financebench Research Agent

# Overview

This repository offers complete examples of implementing GenAI document research agents on Databricks for the [FinanceBench](https://github.com/patronus-ai/financebench). It leverages the Databricks Agent Framework, integrates with [Langchain/Langgraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/), and supports Vector Search, Unity Catalog Functions, and [Genie](https://www.databricks.com/product/ai-bi/genie) — a state-of-the-art text-to-SQL tool developed by Databricks. The agents are deployed using Databricks Model Serving and are monitored through Databricks Model Monitoring. 


# Project Structure

The project is organized into multiple components including notebooks, configs, agents, and environment setup, defined below:
* `setup/document_ingestion_pipeline` contains notebooks and a separate configuration YAML to ingest the FinanceBench PDFs (10ks, 10qs, etc.) in the `setup/data` directory
  * This directory contains notebooks to load the financebench pdfs to a volume, parse the pdfs, extract metadata, clean up the chunks, and create the vector search index
  * `07_ingest_eval_questions` loads the 150 financebench eval questions to a delta table for downstream evaluation
  * `__runner` can be used to execute all the notebooks in sequence
* `agent_config.yaml` should be updated with your models, catalog, schema, etc.
* `01_document_research_agent` contains the core code for the multi-agent supervisor code can be modified to meet customer needs or improve benchmark performance further.
* `02_driver notebook` runs the document agent on example questions, performs the full evaluation, and register/deploys the model. You test any modifications done to `01_document_research_agent` using this notebook.
* `03_eval_agentbricks` is currently WIP. It is used to evaluate the Databricks Knowledge Assistant (KA) Agent Brick on the same financebench benchmark. Instructions to run this on the way.


# Cluster Config

On Databricks, use either a serverless cluster or a standard cluster running Runtime ML 16.4 LTS or higher.

If you’re using a standard Databricks Runtime, please [install](https://docs.databricks.com/aws/en/libraries/cluster-libraries) the required libraries listed in the [requirements.txt](requirements.txt) file. In this case, you can omit the `pip install ...` commands at the beginning of the notebooks.

If you’re using Serverless compute, please uncomment and run the `pip install ...` commands in each notebook to install the necessary libraries.


# Disclaimer

These examples are provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors, copyright holders, or contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

The authors and maintainers of this repository make no guarantees about the suitability, reliability, availability, timeliness, security or accuracy of the software. It is your responsibility to determine that the software meets your needs and complies with your system requirements.

No support is provided with this software. Users are solely responsible for installation, use, and troubleshooting. While issues and pull requests may be submitted, there is no guarantee of response or resolution.

By using this software, you acknowledge that you have read this disclaimer, understand it, and agree to be bound by its terms.
