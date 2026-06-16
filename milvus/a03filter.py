"""
Milvus 元数据过滤示例
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import MilvusClient
# 跟上个示例一样的代码 已经单独分装成一个utils
from milvus.milvus_utils import get_embedding, get_embeddings, API_KEY
# 配置
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_docs_filter"
EMBEDDING_DIM = 4096
def main():
    # 1. 连接 Milvus
    milvus_client = MilvusClient(
uri
=MILVUS_URI)
    # 2. 创建 Collection
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)
    milvus_client.create_collection(

collection_name
=COLLECTION_NAME,

dimension
=EMBEDDING_DIM,

auto_id
=True,

metric_type
="IP",
    )
    print(f"Collection '{COLLECTION_NAME}' created")
    # 3. 准备数据（不同来源）
    documents = [
        "Milvus 是一个开源的向量数据库。",
        "Milvus 使用 etcd 存储元数据。",
        "Python 是一种高级编程语言。",
        "Python 支持多种编程范式。",
        "Java 是面向对象的编程语言。",
    ]
    sources = ["milvus_docs", "milvus_docs", "python_docs", "python_docs", "java_docs"]
    # 4. 插入数据
    embeddings = get_embeddings(documents)
    data = [
        {"vector": embeddings[i], "text": documents[i], "source": sources[i]}
        for i in range(len(documents))
    ]
    milvus_client.insert(COLLECTION_NAME, data)
    milvus_client.load_collection(COLLECTION_NAME)
    print(f"Inserted {len(documents)} documents")
    # 5. 检索（不过滤）
    query = "关于编程语言"
    query_emb = get_embedding(query)
    print("\n--- 全部结果 ---")
    results = milvus_client.search(
        COLLECTION_NAME, [query_emb],
limit
=5,

output_fields
=["text", "source"]
    )
    for hit in results[0]:
        print(f"[{hit['distance']:.4f}] {hit['entity']['text']} (source: {hit['entity']['source']})")
    # 6. 检索（按 source 过滤）
    print("\n--- 过滤: source == 'python_docs' ---")
    results = milvus_client.search(
        COLLECTION_NAME, [query_emb],
limit
=3,

output_fields
=["text", "source"],

filter
='source == "python_docs"'
    )
    for hit in results[0]:
        print(f"[{hit['distance']:.4f}] {hit['entity']['text']}")
    # 7. 检索（多条件过滤）
    print("\n--- 过滤: source in ['milvus_docs', 'java_docs'] ---")
    results = milvus_client.search(
        COLLECTION_NAME, [query_emb],
limit
=3,

output_fields
=["text", "source"],

filter
='source in ["milvus_docs", "java_docs"]'
    )
    for hit in results[0]:
        print(f"[{hit['distance']:.4f}] {hit['entity']['text']}")
    # 8. 纯查询（不需要向量）
    print("\n--- query 方法: source == 'python_docs' ---")
    results = milvus_client.query(
        COLLECTION_NAME,

filter
='source == "python_docs"',

output_fields
=["text", "source"],
    )
    for item in results:
        print(f"- {item['text']} (source: {item['source']})")
    print("\nDone!")
if __name__ == "__main__":
    if not API_KEY:
        print("Error: Please set SILICONFLOW_API_KEY")
        exit(1)
    main()