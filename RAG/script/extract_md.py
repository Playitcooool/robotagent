import os
import shutil

# 新结构源目录
SOURCE_DIR = "output"
# 平铺目标目录
DEST_DIR = "extracted_md"

os.makedirs(DEST_DIR, exist_ok=True)

# 遍历 source_dir
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(".md"):
            source_path = os.path.join(root, file)
            
            # 初始目标路径
            dest_path = os.path.join(DEST_DIR, file)
            
            # 防止文件名重复，增加序号
            count = 1
            base_name, ext = os.path.splitext(file)
            while os.path.exists(dest_path):
                dest_path = os.path.join(DEST_DIR, f"{base_name}_{count}{ext}")
                count += 1
            
            shutil.copy2(source_path, dest_path)
            print(f"已复制: {source_path} -> {dest_path}")

print("所有 Markdown 文件已平铺提取完成！")
