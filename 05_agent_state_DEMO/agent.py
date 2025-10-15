import json
import logging
import os
import time
import urllib.parse
import uuid
from threading import Lock
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    DatabricksFunctionClient,
    UCFunctionToolkit,
)
from databricks.sdk import WorkspaceClient
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from contextlib import contextmanager

from vector_search_utils.self_querying_retriever import load_self_querying_retriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# TODO: Fill in Lakebase config values here
LAKEBASE_CONFIG = {
    "instance_name": "doan-langgraph-memory",
    "conn_host": "instance-d447b6fa-5b0c-43cf-9f4c-81f3e79549fd.database.cloud.databricks.com",
    "conn_db_name": "databricks_postgres",
    "conn_ssl_mode": "require",
}

# TODO make sure you update the config file
configs = mlflow.models.ModelConfig(development_config="./agent_config.yaml")

databricks_config = configs.get('databricks_config')
agent_configs = configs.get("agent_configs")
retriever_config = configs.get('retriever_config')

databricks_config = configs.get('databricks_config')
agent_configs = configs.get("agent_configs")
retriever_config = configs.get('retriever_config')

LLM_ENDPOINT_NAME = agent_configs.get("llm").get("endpoint_name")
LLM_TEMPERATURE = agent_configs.get("llm").get("temperature")
SYSTEM_PROMPT = agent_configs.get("document_agent").get("description")

###############################################################################
## Define tools for your agent,enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT_NAME,
    temperature=LLM_TEMPERATURE,
)

sq_retriever = load_self_querying_retriever(llm, databricks_config, retriever_config)

tools = []

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# List to store vector search tool instances for unstructured retrieval.
VECTOR_SEARCH_TOOLS = [sq_retriever.as_tool()]

# To add vector search retriever tools,
# use VectorSearchRetrieverTool and create_tool_info,
# then append the result to TOOL_INFOS.
# Example:
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )

tools.extend(VECTOR_SEARCH_TOOLS)

#####################
## Define agent logic
#####################


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


class CredentialConnection(psycopg.Connection):
    """Custom connection class that generates fresh OAuth tokens with caching."""
    
    workspace_client = None
    instance_name = None
    
    # Cache attributes
    _cached_credential = None
    _cache_timestamp = None
    _cache_duration = 3000  # 50 minutes in seconds (50 * 60)
    _cache_lock = Lock()
    
    @classmethod
    def connect(cls, conninfo='', **kwargs):
        """Override connect to inject OAuth token with 50-minute caching"""
        if cls.workspace_client is None or cls.instance_name is None:
            raise ValueError("workspace_client and instance_name must be set on CredentialConnection class")
        
        # Get cached or fresh credential and append the new password to kwargs
        credential_token = cls._get_cached_credential()
        kwargs['password'] = credential_token
        
        # Call the superclass's connect method with updated kwargs
        return super().connect(conninfo, **kwargs)
    
    @classmethod
    def _get_cached_credential(cls):
        """Get credential from cache or generate a new one if cache is expired"""
        with cls._cache_lock:
            current_time = time.time()
            
            # Check if we have a valid cached credential
            if (cls._cached_credential is not None and 
                cls._cache_timestamp is not None and 
                current_time - cls._cache_timestamp < cls._cache_duration):
                return cls._cached_credential
            
            # Generate new credential
            credential = cls.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[cls.instance_name]
            )
            
            # Cache the new credential
            cls._cached_credential = credential.token
            cls._cache_timestamp = current_time
            
            return cls._cached_credential


class LangGraphResponsesAgent(ResponsesAgent):
    """Stateful agent using ResponsesAgent with Lakebase PostgreSQL checkpointing.
    
    Features:
    - Connection pooling with credential rotation and caching
    - Thread-based conversation state persistence
    - Tool support with UC functions
    """

    def __init__(self, lakebase_config: dict[str, Any]):
        self.lakebase_config = lakebase_config
        self.workspace_client = WorkspaceClient()
        
        # Model and tools
        self.model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
        self.system_prompt = SYSTEM_PROMPT
        self.model_with_tools = self.model.bind_tools(tools) if tools else self.model
        
        # Connection pool configuration
        self.pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
        self.pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
        self.pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))
        
        # Token cache duration (in minutes, can be overridden via env var)
        cache_duration_minutes = int(os.getenv("DB_TOKEN_CACHE_MINUTES", "50"))
        CredentialConnection._cache_duration = cache_duration_minutes * 60
        
        # Initialize the connection pool with rotating credentials
        self._connection_pool = self._create_rotating_pool()
        
        mlflow.langchain.autolog()

    def _get_username(self) -> str:
        """Get the username for database connection"""
        try:
            sp = self.workspace_client.current_service_principal.me()
            return sp.application_id
        except Exception:
            user = self.workspace_client.current_user.me()
            return user.user_name

    def _create_rotating_pool(self) -> ConnectionPool:
        """Create a connection pool that automatically rotates credentials with caching"""
        # Set the workspace client and instance name on the custom connection class
        CredentialConnection.workspace_client = self.workspace_client
        CredentialConnection.instance_name = self.lakebase_config["instance_name"]
        
        username = self._get_username()
        host = self.lakebase_config["conn_host"]
        database = self.lakebase_config.get("conn_db_name", "databricks_postgres")
        
        # Create pool with custom connection class
        pool = ConnectionPool(
            conninfo=f"dbname={database} user={username} host={host} sslmode=require",
            connection_class=CredentialConnection,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            timeout=self.pool_timeout,
            open=True,
            kwargs={
                "autocommit": True, # Required for the .setup() method to properly commit the checkpoint tables to the database
                "row_factory": dict_row, # Required because the PostgresSaver implementation accesses database rows using dictionary-style syntax
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
        )
        
        # Test the pool
        try:
            with pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            logger.info(
                f"Connection pool with rotating credentials created successfully "
                f"(min={self.pool_min_size}, max={self.pool_max_size}, "
                f"token_cache={CredentialConnection._cache_duration / 60:.0f} minutes)"
            )
        except Exception as e:
            pool.close()
            raise ConnectionError(f"Failed to create connection pool: {e}")
        
        return pool
    
    @contextmanager
    def get_connection(self):
        """Context manager to get a connection from the pool"""
        with self._connection_pool.connection() as conn:
            yield conn
    
    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert from LangChain messages to Responses API format"""
        responses = []
        for message in messages:
            message_dict = message.model_dump()
            msg_type = message_dict["type"]
            
            if msg_type == "ai":
                if tool_calls := message_dict.get("tool_calls"):
                    for tool_call in tool_calls:
                        responses.append(
                            self.create_function_call_item(
                                id=message_dict.get("id") or str(uuid.uuid4()),
                                call_id=tool_call["id"],
                                name=tool_call["name"],
                                arguments=json.dumps(tool_call["args"]),
                            )
                        )
                else:
                    responses.append(
                        self.create_text_output_item(
                            text=message_dict.get("content", ""),
                            id=message_dict.get("id") or str(uuid.uuid4()),
                        )
                    )
            elif msg_type == "tool":
                responses.append(
                    self.create_function_call_output_item(
                        call_id=message_dict["tool_call_id"],
                        output=message_dict["content"],
                    )
                )
            elif msg_type == "human":
                responses.append({
                    "role": "user",
                    "content": message_dict.get("content", "")
                })
        
        return responses
    
    def _create_graph(self, checkpointer: PostgresSaver):
        """Create the LangGraph workflow"""
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "continue"
            return "end"
        
        if self.system_prompt:
            preprocessor = RunnableLambda(
                lambda state: [{"role": "system", "content": self.system_prompt}] + state["messages"]
            )
        else:
            preprocessor = RunnableLambda(lambda state: state["messages"])
        
        model_runnable = preprocessor | self.model_with_tools
        
        def call_model(state: AgentState, config: RunnableConfig):
            response = model_runnable.invoke(state, config)
            return {"messages": [response]}
        
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", RunnableLambda(call_model))
        
        if tools:
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {"continue": "tools", "end": END}
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)
        
        workflow.set_entry_point("agent")
        
        return workflow.compile(checkpointer=checkpointer)

    def _get_or_create_thread_id(self, request: ResponsesAgentRequest) -> str:
        """Get thread_id from request or create a new one.
        
        Priority:
        1. Use thread_id from custom_inputs if present
        2. Use conversation_id from chat context if available
        3. Generate a new UUID
        
        Returns:
            thread_id: The thread identifier to use for this conversation
        """
        ci = dict(request.custom_inputs or {})
        
        if "thread_id" in ci:
            return ci["thread_id"]
        
        # using conversation id from chat context as thread id
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatContext
        if request.context and getattr(request.context, "conversation_id", None):
            return request.context.conversation_id
        
        # Generate new thread_id
        return str(uuid.uuid4())
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction"""
        thread_id = self._get_or_create_thread_id(request)

        ci = dict(request.custom_inputs or {})
        ci["thread_id"] = thread_id
        request.custom_inputs = ci

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs={"thread_id": ci["thread_id"]})
    
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction with PostgreSQL checkpointing"""
        thread_id = self._get_or_create_thread_id(request)
        
        ci = dict(request.custom_inputs or {})
        ci["thread_id"] = thread_id
        request.custom_inputs = ci
        
        # Convert incoming Responses messages to ChatCompletions format
        # LangChain will automatically convert from ChatCompletions to LangChain format
        cc_msgs = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
        langchain_msgs = cc_msgs
        
        checkpoint_config = {"configurable": {"thread_id": thread_id}}
        
        # Use connection from pool
        with self.get_connection() as conn:            
            # Create checkpointer and graph
            checkpointer = PostgresSaver(conn)
            graph = self._create_graph(checkpointer)
            
            # Stream the graph execution
            for event in graph.stream(
                {"messages": langchain_msgs},
                checkpoint_config,
                stream_mode=["updates", "messages"]
            ):
                if event[0] == "updates":
                    for node_data in event[1].values():
                        for item in self._langchain_to_responses(node_data["messages"]):
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item=item
                            )
                # Stream message chunks for real-time text generation
                elif event[0] == "messages":
                    try:
                        chunk = event[1][0]
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            yield ResponsesAgentStreamEvent(
                                **self.create_text_delta(
                                    delta=chunk.content,
                                    item_id=chunk.id
                                ),
                            )
                    except Exception as e:
                        logger.error(f"Error streaming chunk: {e}")


# ----- Export model -----
AGENT = LangGraphResponsesAgent(LAKEBASE_CONFIG)
mlflow.models.set_model(AGENT)
