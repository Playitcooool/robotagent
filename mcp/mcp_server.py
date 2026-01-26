from fastmcp import FastMCP
import mcp

mcp_server = FastMCP("pybullet")


@mcp_server.tool()
def stacking():
    pass


@mcp_server.tool()
def grab_and_place():
    pass
