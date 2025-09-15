# databricks_genai_hackathon

# Overview

This repository offers complete examples of implementing GenAI agents on Databricks. It leverages the Databricks Agent Framework, integrates with [Langchain/Langgraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/), and supports Vector Search, Unity Catalog Functions, and [Genie](https://www.databricks.com/product/ai-bi/genie) — a state-of-the-art text-to-SQL tool developed by Databricks.

The agents are deployed using Databricks Model Serving and are monitored through Databricks Model Monitoring. The repository includes small datasets featuring preprocessed text chunks from publicly available SEC filings, mock structured data for select companies, and an evaluation set of question/answer pairs sourced from the FinanceBench repository. This evaluation data is used for both offline and online performance assessment.


# Project Structure

The project is organized into multiple components including notebooks, configs, agents, and environment setup.

Users have to edit the YAML files for the project and for the individual ChatModels that get deployed on Databricks. The models and the contained agents are specified in notebooks that users can run as is or modify to achieve the specific goals of the hackathon.


# Cluster Config

On Databricks, use either a serverless cluster or a standard cluster running Runtime 15.4 LTS or higher. The Machine Learning Runtime is not recommended.

If you’re using a standard Databricks Runtime, please [install](https://docs.databricks.com/aws/en/libraries/cluster-libraries) the required libraries listed in the [requirements.txt](requirements.txt) file. In this case, you can omit the `pip install ...` commands at the beginning of the notebooks.

If you’re using Serverless compute, please uncomment and run the `pip install ...` commands in each notebook to install the necessary libraries.


# For admins

- ideally, hackathon users should be granted permission to create their individual unity catalog schema. This greatly reduces the need to specify individual assets like tablenames, uc-function names, models etc. 
- caution when cloning the repo to individual users workspace folders: yaml files do not get cloned, users have to copy and edit them manually 

# Project Setup

 - edit [configs/project.yml](configs/project.yml) to specify your settings
   - specify the parameters for project. These parameters are used throughout the repo. It is recommended that individual users/team use their own dedicated `uc_schema`. 
   - In most notebooks the pydantic models in [configs/project.py](configs/project.py) are used to fill in additional parameters. You have the option to specify most of these additional parameters manually in [configs/project.yml](configs/project.yml) but that should typically not be necessary. For example when `source_table_name` is not specified in the yaml file, then it is derived by the pydantic model validator as `<uc_catalog>.<uc_schema>.<table_name>`. 
 - run the project setup notebook  [setup_env/workspace_assets.ipynb](setup_env/workspace_assets.ipynb)
   - for the deployment of the model with a Genie agent, users have to use a Personal Access Token
 - set up the data using the notebooks in the [data](data) folder
 - proceed to the notebooks in the [notebooks](notebooks) folder

 # Notebooks

After the project is configured and the datatables created you can work through the notebooks in the [notebooks](notebooks) folder:
 - [01_create_vector_search_index.ipynb](notebooks/01_create_vector_search_index.ipynb): Create a vector search index based on a delta table that was created in [data/sec_rag_docs_pages.ipynb](data/sec_rag_docs_pages.ipynb)
 - [01_test_vector_search_index.ipynb](notebooks/01_test_vector_search_index.ipynb) test the vector search index
 - [02_create_uc_functions.ipynb](notebooks/02_create_uc_functions.ipynb) create 2 uc functions to use in the chat model
 - [03_synthetic_evals.py](notebooks/03_synthetic_evals.py) this is optional and shows how to generate question/answer pairs based on a delta table holding the prepared content for the vector search.
 - [04_RAG_agent.ipynb](notebooks/04_RAG_agent.ipynb): This notebook demonstrates how to author a LangGraph agent that's compatible with Mosaic AI Agent Framework features:
    - Author a tool-calling LangGraph agent wrapped with `ChatAgent`
    - Manually test the agent's output
    - Evaluate the agent using Mosaic AI Agent Evaluation
    - Log and deploy the agent
    - two companion notebooks are provided for evaluation and monitoring of the deployed chatmodel: [04_RAG_agent_eval.ipynb](notebooks/04_RAG_agent_eval.ipynb) and [04_RAG_agent_monitoring.ipynb](notebooks/04_RAG_agent_monitoring.ipynb)
 - [05_RAG_Genie_agent.ipynb](notebooks/05_RAG_Genie_agent.ipynb): This notebook demonstrates how to build a multi-agent system using Mosaic AI Agent Framework and [LangGraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/), where [Genie](https://www.databricks.com/product/ai-bi/genie) is one of the agents.
    1. Author a multi-agent system using LangGraph.
    2. Wrap the LangGraph agent with MLflow `ChatAgent` to ensure compatibility with Databricks features.
    3. Manually test the multi-agent system's output.
    4. Log and deploy the multi-agent system.

# Disclaimer

These examples are provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors, copyright holders, or contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

The authors and maintainers of this repository make no guarantees about the suitability, reliability, availability, timeliness, security or accuracy of the software. It is your responsibility to determine that the software meets your needs and complies with your system requirements.

No support is provided with this software. Users are solely responsible for installation, use, and troubleshooting. While issues and pull requests may be submitted, there is no guarantee of response or resolution.

By using this software, you acknowledge that you have read this disclaimer, understand it, and agree to be bound by its terms.
