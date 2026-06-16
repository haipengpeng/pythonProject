"""
Milvus CRUD 操作示例
使用 milvus_utils 工具模块实现增删改查功能
"""
from pymilvus import MilvusClient
from milvus_utils import (
    create_collection_if_not_exists,
    get_collection_info,
    show_collections,
    insert_data,
    search_data,
    query_data,
    delete_data_by_ids,
    delete_data_by_filter,
    get_data_by_ids,
    list_indexes,
)
from dotenv import load_dotenv
import requests
import os

# ==================== 配置 ====================
load_dotenv()
API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-lhgpxnfqqjaafeinovhtrqhwauubdrgotbrwkaaybbydzcbs")
API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_docs_bge_m3"
EMBEDDING_DIM = 4096


# ==================== 嵌入函数 ====================
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """调用硅基流动 API 获取文本嵌入"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"input": text, "model": model}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def get_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """批量获取文本嵌入"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"input": texts, "model": model}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    return [item["embedding"] for item in sorted(result["data"], key=lambda x: x["index"])]


# ==================== 示例数据 ====================
SAMPLE_DOCUMENTS = [
    "Milvus 是一个开源的向量数据库，专门用于处理大规模向量相似度搜索。",
    "Milvus 支持多种向量类型，包括 Float32、Float16、BFloat16 和 Binary 向量。",
    "Milvus 的向量数据默认存储在对象存储中，例如 MinIO 或 S3。",
    "Milvus 使用 etcd 存储元数据，例如集合 schema 和索引信息。",
    "BGE-M3 是智谱AI开源的embedding模型，支持Dense、Sparse和ColBERT向量。",
    "硅基流动(SiliconFlow)是一个提供AI模型API服务的平台。",
    "向量数据库在RAG(检索增强生成)系统中扮演着重要角色。",
]
SAMPLE_SOURCES = ["milvus_docs"] * 4 + ["bge_m3_docs"] + ["siliconflow_docs"] + ["vector_db_docs"]


# ==================== CRUD 操作示例 ====================
def demo_create_collection(milvus_client: MilvusClient):
    """演示创建 Collection"""
    print("\n--- 创建 Collection ---")
    create_collection_if_not_exists(
        client=milvus_client,
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIM,
        metric_type="IP",
        auto_id=True,
    )


def demo_show_collections(milvus_client: MilvusClient):
    """演示查看所有 Collection"""
    print("\n--- 查看所有 Collection ---")
    collections = show_collections(client=milvus_client)
    print(f"现有 Collection: {collections}")


def demo_get_collection_info(milvus_client: MilvusClient):
    """演示获取 Collection 信息"""
    print("\n--- Collection 信息 ---")
    info = get_collection_info(client=milvus_client, collection_name=COLLECTION_NAME)
    print(f"Collection 信息: {info}")


def demo_insert_data(milvus_client: MilvusClient):
    """演示插入数据"""
    print("\n--- 插入数据 ---")
    # 生成嵌入向量
    embeddings = get_embeddings(SAMPLE_DOCUMENTS)

    # 准备数据
    data = [
        {
            "vector": embeddings[i],
            "text": SAMPLE_DOCUMENTS[i],
            "source": SAMPLE_SOURCES[i],
        }
        for i in range(len(SAMPLE_DOCUMENTS))
    ]

    # 插入数据
    ids = insert_data(client=milvus_client, collection_name=COLLECTION_NAME, data=data)
    print(f"成功插入 {len(ids)} 条数据，ID: {ids[:3]}...")

    # 加载 Collection
    milvus_client.load_collection(COLLECTION_NAME)
    return ids


def demo_search(milvus_client: MilvusClient):
    """演示向量搜索"""
    print("\n--- 向量搜索 ---")
    query_text = "Milvus 使用什么存储元数据？"
    query_embedding = get_embedding(query_text)

    results = search_data(
        client=milvus_client,
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=3,
        output_fields=["text", "source"],
        search_params={"metric_type": "IP", "params": {}},
    )

    print(f"查询: {query_text}")
    print("搜索结果:")
    for i, hit in enumerate(results[0], 1):
        print(f"  [{i}] 距离: {hit['distance']:.4f}, 文本: {hit['entity']['text'][:50]}...")


def demo_query(milvus_client: MilvusClient):
    """演示条件查询"""
    print("\n--- 条件查询 ---")
    # 查询 source 为 "milvus_docs" 的数据
    results = query_data(
        client=milvus_client,
        collection_name=COLLECTION_NAME,
        filter="source == 'milvus_docs'",
        output_fields=["text", "source"],
        limit=10,
    )
    print(f"查询 source=='milvus_docs' 的结果: {len(results)} 条")
    for r in results[:2]:
        print(f"  - {r.get('text', '')[:50]}...")


def demo_get_by_ids(milvus_client: MilvusClient, ids: list):
    """演示根据 ID 获取数据"""
    print("\n--- 根据 ID 查询 ---")
    if not ids:
        print("没有数据 ID，跳过")
        return

    test_ids = ids[:3]
    results = get_data_by_ids(
        client=milvus_client,
        collection_name=COLLECTION_NAME,
        ids=test_ids,
        output_fields=["text", "source"],
    )
    print(f"根据 ID 查询结果: {len(results)} 条")
    for r in results:
        print(f"  - ID: {r.get('id')}, text: {r.get('text', '')[:30]}...")


def demo_delete_by_ids(milvus_client: MilvusClient, ids: list):
    """演示根据 ID 删除数据"""
    print("\n--- 根据 ID 删除 ---")
    if not ids or len(ids) < 2:
        print("数据不足，跳过删除")
        return

    delete_ids = ids[:1]  # 只删除第一条
    count_before = len(query_data(milvus_client, COLLECTION_NAME, output_fields=["id"]))

    delete_data_by_ids(
        client=milvus_client,
        collection_name=COLLECTION_NAME,
        ids=delete_ids,
    )

    count_after = len(query_data(milvus_client, COLLECTION_NAME, output_fields=["id"]))
    print(f"删除了 {len(delete_ids)} 条数据，删除前: {count_before}, 删除后: {count_after}")


def demo_delete_by_filter(milvus_client: MilvusClient):
    """演示根据条件删除数据"""
    print("\n--- 根据条件删除 ---")
    # 删除 source 为 "bge_m3_docs" 的数据
    count_before = len(query_data(
        milvus_client, COLLECTION_NAME,
        filter="source == 'bge_m3_docs'",
        output_fields=["id"]
    ))

    if count_before > 0:
        delete_data_by_filter(
            client=milvus_client,
            collection_name=COLLECTION_NAME,
            filter="source == 'bge_m3_docs'",
        )
        print(f"删除了 {count_before} 条 source=='bge_m3_docs' 的数据")
    else:
        print("没有找到需要删除的数据")


def demo_list_indexes(milvus_client: MilvusClient):
    """演示查看索引"""
    print("\n--- 查看索引 ---")
    indexes = list_indexes(client=milvus_client, collection_name=COLLECTION_NAME)
    print(f"索引列表: {indexes}")


# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("Milvus CRUD 操作示例")
    print("=" * 50)

    # 1. 连接 Milvus
    print("\n[1] 连接 Milvus...")
    milvus_client = MilvusClient(uri=MILVUS_URI)
    print("    Milvus 连接成功!")

    # 2. 清理旧数据（可选）
    print("\n[2] 清理旧 Collection...")
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)
        print(f"    已删除旧的 Collection: {COLLECTION_NAME}")

    # 3. 创建 Collection
    demo_create_collection(milvus_client)

    # 4. 查看所有 Collection
    demo_show_collections(milvus_client)

    # 5. Collection 信息
    demo_get_collection_info(milvus_client)

    # 6. 插入数据
    inserted_ids = demo_insert_data(milvus_client)

    # 7. 向量搜索
    demo_search(milvus_client)

    # 8. 条件查询
    demo_query(milvus_client)

    # 9. 根据 ID 查询
    demo_get_by_ids(milvus_client, inserted_ids)

    # 10. 根据 ID 删除
    demo_delete_by_ids(milvus_client, inserted_ids)

    # 11. 根据条件删除
    demo_delete_by_filter(milvus_client)

    # 12. 查看索引
    demo_list_indexes(milvus_client)

    print("\n" + "=" * 50)
    print("CRUD 示例完成!")
    print("=" * 50)


if __name__ == "__main__":
    if API_KEY == "your-api-key-here":
        print("错误: 请先设置硅基流动 API Key!")
        exit(1)
    main()