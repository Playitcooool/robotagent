from fastmcp import FastMCP
from langchain_core.tools import tool


@tool(description="Use this tool to calculate the sum of two numbers")
def sum_two_num(a: int, b: int):
    return a + b


@tool(description="Use this tool to calculate the product of two numbers")
def product_two_num(a: int, b: int):
    return a * b
