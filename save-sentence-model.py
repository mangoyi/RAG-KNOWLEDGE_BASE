# save_model.py
from sentence_transformers import SentenceTransformer

# 指定模型名称和你想保存的本地路径
model_name = "sentence-transformers/all-MiniLM-L6-v2"
local_model_path = "./models/all-MiniLM-L6-v2" # 可自定义路径

# 下载并保存模型
print(f"正在下载模型 {model_name} 到 {local_model_path}...")
model = SentenceTransformer(model_name)
model.save(local_model_path)
print("模型下载并保存完成！")