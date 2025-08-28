from dotenv import load_dotenv
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio
import os
from datetime import datetime
import asyncio

load_dotenv(override=True)

async def main():
    # ----------------------------
    # 1️⃣ Agent mémoire persistante (LibSQL)
    # The first type of MCP Server: runs locally, everything local
    # ----------------------------
    params_libsql = {
        "command": "npx",
        "args": ["-y", "mcp-memory-libsql"],
        "env": {"LIBSQL_URL": "file:./memory/vincent.db"}
    }

    async with MCPServerStdio(params=params_libsql, client_session_timeout_seconds=30) as libsql_server:
        agent_libsql = Agent(
            name="libsql_agent",
            instructions="You use your entity tools as a persistent memory to store and recall information about your conversations.",
            model="gpt-4o-mini",
            mcp_servers=[libsql_server]
        )

        # Exemple 1
        with trace("conversation"):
            result = await Runner.run(agent_libsql, "My name's Vincent. I'm an LLM engineer.")
        print(result.final_output)

        # Exemple 2
        with trace("conversation"):
            result2 = await Runner.run(agent_libsql, "My name's Vincent. What do you know about me?")
        print(result2.final_output)

    # ----------------------------
    # 2️⃣ Agent recherche web (Brave Search)
    # The 2nd type of MCP server - runs locally, calls a web service
    # ----------------------------
    env_brave = {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
    params_brave = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": env_brave
    }

    async with MCPServerStdio(params=params_brave, client_session_timeout_seconds=30) as brave_server:
        agent_brave = Agent(
            name="brave_agent",
            instructions="You are able to search the web for information and briefly summarize the takeaways.",
            model="gpt-4o-mini",
            mcp_servers=[brave_server]
        )

        request = f"Find the top 3 most interesting AI breakthroughs reported this week and summarize each in 2-3 sentences. \
Include the source and any relevant links. Current date: {datetime.now().strftime('%Y-%m-%d')}"


        with trace("conversation"):
            result = await Runner.run(agent_brave, request)
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())