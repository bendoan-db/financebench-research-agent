# Databricks GenAI Hackathon Smoke Test
This directory contains notebooks to test for the functionalities required to execute the GenAI hackathon. This test will create a vector search endpoint, a vector search index, and a model serving endpoint. Be sure to delete the resources after running to save on costs.

## Prequisites
- Review the Databricks Agent Framework documentation

## Usage Instructions
1. Update `smoke_test_config.yaml` with the correct parameters
2. Select Run All for the 00_runner notebook


## Compute Requirements
This test should be run on a standard cluster running DBR 16.4 ML LTS