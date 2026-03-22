import chainlit as cl
from config import PostgreSettings
from settings.chat_settings import ChatSettings
from profiles.chat_profiles import ChatProfiles
from core.essentials import CoreEssentials
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools


chat_settings = ChatSettings().get_chat_settings()
chat_profiles = ChatProfiles().get_chat_profiles()
core = CoreEssentials()
postgre_settings = PostgreSettings()


@cl.on_mcp_connect 
async def on_mcp(connection, session: ClientSession):
    result = await load_mcp_tools(session)
    cl.user_session.set("mcp_tools",result)
    

@cl.data_layer  
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=postgre_settings.POSTGRE_CONNECTION_STRING, storage_provider=None
    )


@cl.on_chat_resume  
async def resume_conversation(thread: ThreadDict):
    settings = thread.get("metadata").get("chat_settings")  # type: ignore
    await core.setup_chat(settings["Model"], settings["Temperature"])  # type: ignore
    await cl.ChatSettings(chat_settings).send() 


@cl.password_auth_callback   # type: ignore
def auth_callback(username: str, password: str):
    if username == "prospectio" and password == "prospectio":
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        ) 
    return None


@cl.set_chat_profiles   # type: ignore
async def chat_profile():
    return chat_profiles


@cl.on_chat_start
async def on_chat_start():
    await core.connect_mcp_for_session()
    await core.setup_chat(chat_settings[0].values[0], chat_settings[1].initial)  # type: ignore
    await cl.ChatSettings(chat_settings).send() 


@cl.on_message  
async def main(msg: cl.Message): 
    try:
        response = await core.call_agent()
        await core.process_response(response)
    except Exception as e:
        await cl.Message(content=f"{type(e).__name__}: {e}").send()  


@cl.on_settings_update  
async def setup_agent(settings):
    await core.setup_chat(settings["Model"], settings["Temperature"])  