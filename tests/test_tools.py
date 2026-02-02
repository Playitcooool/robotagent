import asyncio
from fastmcp import Client

# MCP Server 地址
client = Client("http://localhost:8001/mcp")


async def test_tools():
    async with client:
        # ======================
        # 1. stacking
        # ======================
        stacking_result = await client.call_tool(
            "stacking", {"args": {"gui": False, "settle_steps": 240}}
        )
        print("stacking result:", stacking_result)

        # ======================
        # 2. grab_and_place
        # ======================
        grab_result = await client.call_tool(
            "grab_and_place",
            {
                "args": {
                    "gui": False,
                    "start_position": [0.2, 0.0, 0.02],
                    "target_position": [0.4, 0.4, 0.02],
                }
            },
        )
        print("grab_and_place result:", grab_result)

        # ======================
        # 3. path_tracking
        # ======================
        path_result = await client.call_tool(
            "path_tracking", {"args": {"gui": False, "radius": 0.3, "steps": 60}}
        )
        print("path_tracking result:", path_result)

        # ======================
        # 4. push_cube
        # ======================
        push_result = await client.call_tool(
            "push_cube",
            {
                "args": {
                    "gui": False,
                    "start_position": [0, 0, 0.02],
                    "push_vector": [0.2, 0, 0],
                    "steps": 60,
                }
            },
        )
        print("push_cube result:", push_result)

        # ======================
        # 5. pick_and_throw
        # ======================
        throw_result = await client.call_tool(
            "pick_and_throw",
            {
                "args": {
                    "gui": False,
                    "start_position": [0, 0, 0.02],
                    "throw_vector": [0.3, 0.3, 0.2],
                    "settle_steps": 60,
                }
            },
        )
        print("pick_and_throw result:", throw_result)


if __name__ == "__main__":
    asyncio.run(test_tools())
