import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.documents import Document


# 连接 Qdrant
client = QdrantClient(host="localhost", port=6333)

# 与写入时一致的 embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 从已有 collection 创建向量库对象
qdrant_store = QdrantVectorStore(
    client=client,
    collection_name="markdown_chunks",
    embedding=embeddings
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    query = embeddings.embed_query(query)
    retrieved_docs = client.query_points(query=query, 
                                         collection_name='markdown_chunks',
                                         limit=3)

    documents = []
    serialized_parts = []

    # Qdrant's response may come in different shapes (tuple, dict, or object).
    # Normalize to an iterable of scored points called `points`.
    points = None
    if isinstance(retrieved_docs, dict):
        points = retrieved_docs.get("points") or retrieved_docs.get("result") or []
    elif isinstance(retrieved_docs, (tuple, list)) and len(retrieved_docs) >= 2 and isinstance(retrieved_docs[1], list):
        # e.g. ('points', [ScoredPoint(...), ...])
        points = retrieved_docs[1]
    elif hasattr(retrieved_docs, "points"):
        points = getattr(retrieved_docs, "points")
    elif hasattr(retrieved_docs, "result"):
        points = getattr(retrieved_docs, "result")
    else:
        # Fallback: assume it's already an iterable of points
        points = retrieved_docs

    for doc in points:
        # ScoredPoint objects expose `.payload`; sometimes doc can be a tuple/indexed structure.
        if hasattr(doc, "payload"):
            metadata = doc.payload or {}
        elif isinstance(doc, (tuple, list)) and len(doc) >= 2 and hasattr(doc[1], "payload"):
            metadata = doc[1].payload or {}
        elif isinstance(doc, dict):
            metadata = doc.get("payload") or {}
        else:
            metadata = {}

        text = ""
        if isinstance(metadata, dict):
            text = metadata.get("text", "")
        elif isinstance(metadata, str):
            text = metadata

        if not text:
            print("[WARN] No text in metadata:", metadata)
            continue

        documents.append(Document(page_content=text, metadata=metadata))
        serialized_parts.append(f"Source: {metadata}\nContent: {text}")

    serialized = "\n\n".join(serialized_parts)

    if not serialized:
        print("[WARN] No chunks retrieved for query:", query)

    return serialized, documents





tools = [retrieve_context]

sys_prompt = (
    "You have access to a tool that retrieves paper from vector database. "
    "Use the tool to help answer user queries."
)

llm = ChatOllama(
    model='qwen3:1.7b',
    base_url='http://localhost:11434',
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt= sys_prompt
)

query = (
    "<AlignSurvey: A Comprehensive Benchmark for Human Preferences Alignment in Social Surveys>这篇论文的研究方法是什么，请详细说说"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()