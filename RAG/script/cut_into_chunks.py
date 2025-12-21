import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 原 Markdown 目录
MD_DIR = "RAG/extracted_md"
# 切分后的输出目录
CHUNK_DIR = "RAG/chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)

# 切分器配置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,       # 每个 chunk 大小，可按需调整
    chunk_overlap=200,     # 重叠部分
    separators=["\n\n", "\n", " ", ""]
)

# 遍历 Markdown 文件并切分
for file_name in os.listdir(MD_DIR):
    if file_name.endswith(".md"):
        file_path = os.path.join(MD_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = text_splitter.split_text(text)
        
        # 保存每个 chunk 为单独文件
        base_name = os.path.splitext(file_name)[0]
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(CHUNK_DIR, f"{base_name}_chunk_{i}.md")
            with open(chunk_file, "w", encoding="utf-8") as cf:
                cf.write(chunk)
            print(f"保存: {chunk_file}")

print("Markdown 文件切分并保存完成！")
