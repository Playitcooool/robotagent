from fastmcp import FastMCP
import mcp

mcp_server = FastMCP("pybullet")


@mcp.tool()
def stacking():
    pass


@mcp.tool()
def grab_and_place():
    pass
