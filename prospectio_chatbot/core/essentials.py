from typing import Any, AsyncIterator
from config import MCPSettings
import chainlit as cl
from graphs.generic_graph import GenericGraph
from langchain_core.messages import AIMessageChunk
from graphs.graph_factory import GraphFactory
from graphs.graph_params import GraphParams
from langchain_core.runnables import RunnableConfig
from chainlit.types import ConnectSseMCPRequest
from chainlit.server import connect_mcp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoreEssentials:

    nodes_mapping = {
        "Prospectio": "call_model",
    }

    def __init__(self):
        self.graph_params = GraphParams()
        self.graph_factory = GraphFactory(self.graph_params)
        self.mcp_servers = MCPSettings().MCP_SERVERS

    async def setup_chat(self, model: str, temperature: float):
        # Copilot
        self.graph_params.agent = (
            cl.user_session.get("chat_profile") or "Prospectio"
        )
        self.graph_params.model = model
        self.graph_params.temperature = temperature
        cl.user_session.set("model", model)
        cl.user_session.set("temperature", temperature)
        cl.user_session.set("graph", self.graph_factory.create_graph())

    async def call_agent(self) -> AsyncIterator[dict[str, Any] | Any]:
        tools_list = cl.user_session.get("mcp_tools") or []
        self.graph_params.tools_list = tools_list
        graph: GenericGraph = self.graph_factory.create_graph()
        chat_history = cl.chat_context.to_openai()
        config = {"recursion_limit": 100,"configurable": {"thread_id": cl.context.session.id},}
        cb = cl.LangchainCallbackHandler()
        response = graph.get_graph().astream(
            {"messages": chat_history},
            stream_mode=["messages", "updates"],
            config=RunnableConfig(callbacks=[cb], **config),
        )
        return response

    async def process_response(self, response: AsyncIterator[dict[str, Any] | Any]):
        answer = cl.Message(content="")
        # Copilot
        node = cl.user_session.get("chat_profile") or "Prospectio"
        node_name = self.nodes_mapping[node]
        final_node = node_name.split(",")[0]
        sources_node = (
            node_name.split(",")[1] if len(node_name.split(",")) > 1 else None
        )
        settings = f"{cl.user_session.get('chat_profile')} - Model : {cl.user_session.get('model')} - Temperature : {cl.user_session.get('temperature')}\n"
        answer.elements.append(cl.Text(content=f"{settings}", display="inline")) # type: ignore
        async for chunk in response:
            if chunk[0] == "messages" and chunk[1][1]["langgraph_node"] == final_node: # type: ignore
                values: AIMessageChunk = chunk[1][0] # type: ignore
                await answer.stream_token(values.content) # type: ignore
            if node_name:
                await self.process_sources(sources_node, chunk, answer) # type: ignore
        await answer.send()

    async def process_sources(self, node_name: str, chunk, answer: cl.Message) -> str:
        sources = []
        if chunk[0] == "updates":
            if node_name in chunk[1] and "sources" in chunk[1][node_name]:
                sources = chunk[1][node_name]["sources"]
                formatted_sources = "\n".join(sources)
                sources = f"Sources:\n{formatted_sources}"
                answer.elements.append(cl.Text(content=f"{sources}", display="inline")) # type: ignore
        return sources # type: ignore
    
    async def connect_mcp_for_session(self) -> dict:
        """
        Asynchronously connects to all MCP servers defined in the configuration for the current session.
    
        Iterates through the list of MCP servers, creates a connection request for each, and attempts to establish a connection
        using the current user's session context. Returns a success message if all connections are successful, otherwise logs
        the error and returns a failure message with error details.
    
        Returns:
            dict: A dictionary containing a success or failure message, and error details if an exception occurred.
        """
        try:
            for mcp in self.mcp_servers:
                mcp_request = ConnectSseMCPRequest(
                    sessionId=cl.context.session.id,
                    clientType=mcp["clientType"],
                    name=mcp["name"],
                    url=mcp["url"],
                )
                await connect_mcp(payload=mcp_request, current_user=cl.context.session.user)
            return {"message": "Connected to MCP servers successfully."}
        except Exception as e:
            logger.error(f"Error connecting to MCP servers: {e}")
            return {"message": "Failed to connect to MCP servers.", "error": str(e)}
