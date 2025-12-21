import os
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------
# 配置
# ----------------------
CHUNK_DIR = "RAG/chunks"
QDRANT_COLLECTION_NAME = "markdown_chunks"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
BATCH_SIZE = 256

# ----------------------
# 初始化 Qdrant 客户端
# ----------------------
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------
# 读取 chunk
# ----------------------
texts = []
metadatas = []

for file_name in os.listdir(CHUNK_DIR):
    if file_name.endswith(".md"):
        path = os.path.join(CHUNK_DIR, file_name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        texts.append(content)
        metadatas.append({
            "id": str(uuid.uuid4()),
            "source": file_name,
            "n_tokens": len(content.split())
        })

print(f"加载 {len(texts)} 个 chunk")

# ----------------------
# 计算 embeddings
# ----------------------
vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)

# ----------------------
# 创建 Qdrant collection
# ----------------------
client.recreate_collection(
    collection_name=QDRANT_COLLECTION_NAME,
    vectors_config=VectorParams(
        size=vectors.shape[1],
        distance=Distance.COSINE
    )
)

# ----------------------
# 分批写入 Qdrant（文本直接存 payload）
# ----------------------
batch = []
for idx, meta in enumerate(metadatas):
    batch.append(
        PointStruct(
            id=meta["id"],
            vector=vectors[idx].tolist(),
            payload={
                "source": meta["source"],
                "text": texts[idx],
                "n_tokens": meta["n_tokens"]
            }
        )
    )
    if len(batch) >= BATCH_SIZE:
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch)
        batch = []

if batch:
    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch)

print(f"已将 {len(metadatas)} 条向量和文本写入 Qdrant")
