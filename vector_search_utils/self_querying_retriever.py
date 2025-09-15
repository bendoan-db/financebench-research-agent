from databricks.vector_search.client import VectorSearchClient

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.databricks_vector_search import DatabricksVectorSearchTranslator

from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    load_query_constructor_runnable,
)

from databricks_langchain import DatabricksVectorSearch

def load_self_querying_retriever(model, databricks_config, retriever_config):
    """
    Loads selfquerying retriever using the retriever config and model

    Args:
        model: LLM to use with retriever
        retriever config: Retriever configuration dictionary

    Returns:
        multiquery_retriever: Langchain multiquery retriever object
    """

    vector_search_schema = retriever_config.get("schema")

    metadata_field_info = [
        AttributeInfo(
            name="resolved_company",
            description="""The name of the company being analyzed. Names should be FULLY capitalized. Do not use hypens or underscores. Values can ONLY BE ONE OF THE FOLLOWING:
            '3M',
            'ACTIVISIONBLIZZARD',
            'ADOBE',
            'AES',
            'AMAZON',
            'AMCOR',
            'AMD',
            'AMERICANEXPRESS',
            'AMERICANWATERWORKS',
            'APPLE',
            'BESTBUY',
            'BLOCK',
            'BOEING',
            'BOSTONPROPERTIES',
            'COCACOLA',
            'CORNING',
            'COSTCO',
            'CVSHEALTH',
            'EBAY',
            'FEDEX',
            'FOOTLOCKER',
            'GENERALMILLS',
            'INTEL',
            'JOHNSONJOHNSON',
            'JPMORGAN',
            'KRAFTHEINZ',
            'LOCKHEEDMARTIN',
            'MCDONALDS',
            'MGMRESORTS',
            'MICROSOFT',
            'NETFLIX',
            'NIKE',
            'ORACLE',
            'PAYPAL',
            'PEPSICO',
            'PFIZER',
            'PG&E',
            'SALESFORCE',
            'ULTABEAUTY',
            'VERIZON',
            'WALMART'
            """,
            type="string"
        ),
        AttributeInfo(
            name="document_type",
            description="The type of SEC document. Values can be one of the following: '8k', '10q', 'Earnings Report', '10k'",
            type="string",
        ),
        AttributeInfo(
            name="year", description="Year of the document REPRESENTED AS A STRING", type="string"
            ),
    ]

    examples = [
        (
            "What is Netflix's year end FY2017 total current liabilities (in USD millions)? Base your judgments on the information provided primarily in the balance sheet.",
            {
                "query": "Netflix 2017 total current liabilities",
                "filter": 'and(eq("resolved_company", "NETFLIX"), eq("year", 2017))',
            },
        ),
        (
            "What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision Blizzard? Answer in units of percents and round to one decimal place. Calculate (or extract) the answer from the statement of income and the cash flow statement.",
            {
                "query": "FY2017 FY2018 FY2019 average capex Activision Blizzard",
                "filter": 'and(eq("resolved_company", "ACTIVISIONBLIZZARD"), gte("year", 2017), lte("year", 2019))',
            },
        ),
        (
            "What was the key agenda of the AMCOR's 8k filing dated 1st July 2022?",
            {
                "query": "2022 amcor 8k key agenda",
                "filter": 'and(eq("resolved_company", "AMCOR"), eq("year", 2022))',
            },
        )
    ]

    # Turn the Vector Search index into a LangChain retriever
    vector_store = DatabricksVectorSearch(
        endpoint=retriever_config.get("vector_search_endpoint"),
        index_name=f"{databricks_config.get('catalog')}.{databricks_config.get('schema')}.{retriever_config.get('vector_search_index')}",
        columns=[
            vector_search_schema.get("primary_key"),
            vector_search_schema.get("chunk_text"),
            vector_search_schema.get("document_uri"),
            vector_search_schema.get("doc_year"),
            vector_search_schema.get("doc_name"),
        ],
    )

    doc_contents = "Detailed financial statements, balance sheets, and information that are contained in the SEC documents."
    query_constructor = load_query_constructor_runnable(
        model, doc_contents, metadata_field_info, examples=examples, fix_invalid=True
    )

    self_query_retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vector_store,
        structured_query_translator=DatabricksVectorSearchTranslator(),
        search_kwargs= {
        "k":10,
        "query_type": "hybrid"
        }
    )

    return self_query_retriever