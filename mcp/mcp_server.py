from fastmcp import FastMCP

mcp_server = FastMCP("pybullet")


@mcp.tool()
def stacking():
    pass


@mcp.tool()
def grab_and_place():
    pass
