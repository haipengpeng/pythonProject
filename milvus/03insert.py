"""
Milvus + SiliconFlow 嵌入示例
使用硅基流动的 Qwen/Qwen3-VL-Embedding-0.6B 模型进行嵌入和检索
"""
from pymilvus import MilvusClient
import requests
from dotenv import load_dotenv
import os
# ==================== 配置 ====================
# 硅基流动 API Key (从 https://siliconflow.cn 获取)
# 可以设置环境变量 SILICONFLOW_API_KEY （强烈推荐）或直接在这里填入
load_dotenv()
API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-lhgpxnfqqjaafeinovhtrqhwauubdrgotbrwkaaybbydzcbs")
# 硅基流动 API 地址
API_URL = "https://api.siliconflow.cn/v1/embeddings"
# 嵌入模型，对应 curl 中的 model
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-0.6B"
# Milvus 连接地址
MILVUS_URI = "http://localhost:19530"
# Collection 名称
COLLECTION_NAME = "rag_docs_bge_m3"
# Qwen3-VL-Embedding-0.6B 向量维度
EMBEDDING_DIM = 1024
# ==================== 嵌入函数 ====================
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    调用硅基流动 API 获取文本嵌入
    Args:
        text: 要嵌入的文本
        model: 模型名称
    Returns:
        嵌入向量列表
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": model,
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result["data"][0]["embedding"]


def get_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    批量获取文本嵌入
    Args:
        texts: 要嵌入的文本列表
        model: 模型名称
    Returns:
        嵌入向量列表
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": texts,
        "model": model,
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    # 按输入顺序返回嵌入向量
    embeddings = [item["embedding"] for item in sorted(result["data"], key=lambda x: x["index"])]
    return embeddings
# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("Milvus + SiliconFlow 示例")
    print("=" * 50)
    # 1. 连接 Milvus
    print("\n[1] 连接 Milvus...")
    milvus_client = MilvusClient(uri=MILVUS_URI)
    print("    Milvus 连接成功!")
    # 2. 如果 Collection 已存在，先删除
    print(f"\n[2] 检查 Collection: {COLLECTION_NAME}")
    if milvus_client.has_collection(COLLECTION_NAME):
        print("    Collection 已存在，删除并重建...")
        milvus_client.drop_collection(COLLECTION_NAME)
    else:
        print("    Collection 不存在，将创建新的")
    # 3. 创建 Collection
    print("\n[3] 创建 Collection...")
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIM,
        auto_id=True,metric_type="IP",  # 内积度量，适合归一化向量
        consistency_level="Bounded",
    )
    print(f"    Collection '{COLLECTION_NAME}' 创建成功!")
    # 4. 准备要插入的文档数据
    print("\n[4] 准备文档数据...")
    documents = [
        "Milvus 是一个开源的向量数据库，专门用于处理大规模向量相似度搜索。",
        "Milvus 支持多种向量类型，包括 Float32、Float16、BFloat16 和 Binary 向量。",
        "Milvus 的向量数据默认存储在对象存储中，例如 MinIO 或 S3。",
        "Milvus 使用 etcd 存储元数据，例如集合 schema 和索引信息。",
        "BGE-M3 是智谱AI开源的embedding模型，支持Dense、Sparse和ColBERT向量。",
        "硅基流动(SiliconFlow)是一个提供AI模型API服务的平台。",
        "向量数据库在RAG(检索增强生成)系统中扮演着重要角色。",
    ]
    # 添加来源信息
    sources = ["milvus_docs"] * 4 + ["bge_m3_docs"] + ["siliconflow_docs"] + ["vector_db_docs"]
    print(f"    共 {len(documents)} 条文档")
    # 5. 生成嵌入向量
    print("\n[5] 调用硅基流动 API 生成嵌入向量...")
    print(f"    使用模型: {EMBEDDING_MODEL}")
    embeddings = get_embeddings(documents)
    print(f"    生成 {len(embeddings)} 个向量，每个维度: {len(embeddings[0])}")
    # 6. 插入数据到 Milvus
    print("\n[6] 插入数据到 Milvus...")
    data = [
        {
            "vector": embeddings[i],
            "text": documents[i],
            "source": sources[i],
        }
        for i in range(len(documents))
    ]
    insert_result = milvus_client.insert(
        collection_name=COLLECTION_NAME,data=data
    )
    print(f"    插入成功! 插入 ID 数量: {len(insert_result['ids'])}")
    # 7. 加载 Collection
    print("\n[7] 加载 Collection 到内存...")
    milvus_client.load_collection(COLLECTION_NAME)
    print("    加载成功!")
    # 8. 执行检索
    print("\n[8] 执行检索测试...")
    # 测试查询
    test_queries = [
        "Milvus 使用什么存储元数据？",
        "BGE-M3 模型有什么特点？",
    ]
    for query in test_queries:
        print(f"\n    查询: {query}")
        # 生成查询向量
        query_embedding = get_embedding(query)
        # 搜索
        search_results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=3,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text", "source"],
        )
        print("    检索结果:")
        for i, hit in enumerate(search_results[0], 1):
            print(f"      [{i}] 距离: {hit['distance']:.4f}")
            print(f"          文本: {hit['entity']['text']}")
            print(f"          来源: {hit['entity']['source']}")
    print("\n" + "=" * 50)
    print("示例完成!")
    print("=" * 50)
if __name__ == "__main__":
    # 检查 API Key
    if API_KEY == "your-api-key-here":
        print("错误: 请先设置硅基流动 API Key!")
        print("方法1: 设置环境变量 SILICONFLOW_API_KEY")
        print("方法2: 修改脚本中的 API_KEY 变量")
        exit(1)
    main()
