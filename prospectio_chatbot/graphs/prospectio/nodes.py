from graphs.prospectio.chains.generate import GenerateChain
from langgraph.graph import END, MessagesState
from langgraph.graph import MessagesState
from graphs.graph_params import GraphParams
from prompts.prompt_loader import PromptLoader
from langgraph.prebuilt import ToolNode
from langchain_core.messages.base import BaseMessage


class ProspectioNodes:

    prompt_loader = PromptLoader()

    def __init__(self, graph_params: GraphParams):
        self.graph_params = graph_params
        self.tool_node = ToolNode(graph_params.tools_list)
        self.generate_chain = GenerateChain(
            model=graph_params.model,
            temperature=graph_params.temperature,
            prompt=self.prompt_loader.load_prompt(graph_params.agent),
            tools_list=graph_params.tools_list,
        )

    def should_continue(self, state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    async def call_tools(self, state: MessagesState):
        messages = state["messages"]
        response = await self.tool_node.ainvoke({"messages": messages})
        return response

    async def call_model(self, state: MessagesState):
        messages = state["messages"]
        response = await self.generate_chain.chain.ainvoke({"messages": messages})
        return {"messages": [response]}
