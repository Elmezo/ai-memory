import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def session_1():
    print("--> [Session 1] Connecting to MCP Server via stdio...")
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("--> [Session 1] MCP Initialized.")
            res = await session.call_tool("memory_save", arguments={
                "user_id": "user_ar_mcp",
                "tenant_id": "tenant_mcp",
                "agent_id": "ar_support_bot",
                "project_id": "crm_system",
                "type": "preference",
                "content": "تفضيل العميل: إصدار التقارير المحاسبية باللغة العربية كل نهاية شهر.",
                "tags": ["تقارير", "محاسبة", "عربي"]
            })
            data = json.loads(res.content[0].text)
            mem_id = data.get("memory", {}).get("memory_id")
            print(f"--> [Session 1] Saved status: {data.get('status')} | ID: {mem_id}")
            return mem_id

async def session_2(saved_id):
    print("--> [Session 2] Starting completely new MCP client process...")
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("--> [Session 2] MCP Initialized.")
            
            # 1. Retrieve by exact memory_id
            res_get = await session.call_tool("memory_get", arguments={
                "memory_id": saved_id,
                "user_id": "user_ar_mcp",
                "tenant_id": "tenant_mcp"
            })
            got = json.loads(res_get.content[0].text)
            print(f"--> [Session 2] Successfully retrieved memory: ID={got.get('memory_id')}")
            
            # 2. Search context by Arabic query
            res_search = await session.call_tool("memory_search", arguments={
                "user_id": "user_ar_mcp",
                "tenant_id": "tenant_mcp",
                "query": "التقارير المحاسبية"
            })
            search_data = json.loads(res_search.content[0].text)
            print(f"--> [Session 2] Search hits count: {search_data.get('count')}")
            for m in search_data.get("memories", []):
                print(f"--> [Session 2] Hit ID={m.get('memory_id')}, Agent={m.get('agent_id')}, Project={m.get('project_id')}")

async def main():
    saved_id = await session_1()
    await session_2(saved_id)

if __name__ == "__main__":
    asyncio.run(main())
