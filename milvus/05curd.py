"""
Milvus CRUD 操作示例：查询、删除、加载、删除 Collection
"""

from pymilvus import MilvusClient
from milvus_utils import get_embedding, get_embeddings, API_KEY

# 配置
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "crud_demo"
EMBEDDING_DIM = 1024


def main():
    # 1. 连接 Milvus
    milvus_client = MilvusClient(uri=MILVUS_URI)

    # 2. 创建 Collection 并插入测试数据
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)

    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIM,
        auto_id=True,
        metric_type="IP",
    )

    documents = [
        "Milvus 是向量数据库。",
        "Python 是一种编程语言。",
        "向量数据库用于 AI。",
    ]

    sources = ["tech", "language", "tech"]

    embeddings = get_embeddings(documents)

    data = [
        {
            "vector": embeddings[i],
            "text": documents[i],
            "source": sources[i],
        }
        for i in range(len(documents))
    ]

    milvus_client.insert(COLLECTION_NAME, data)
    milvus_client.load_collection(COLLECTION_NAME)

    print(
        f"Created collection '{COLLECTION_NAME}' "
        f"with {len(documents)} documents\n"
    )

    # ========= 查询操作 =========
    print("=" * 50)
    print("1. 查询：按条件检索实体")
    print("=" * 50)

    # 通过过滤表达式查询
    print("\n--- 查询 source == 'tech' ---")

    results = milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter='source == "tech"',
        output_fields=["id", "text", "source"],
    )

    for item in results:
        print(
            f"id: {item['id']}, "
            f"text: {item['text']}, "
            f"source: {item['source']}"
        )

    # 通过主键查询（需要先获取 id）
    ids = [r["id"] for r in results]

    if ids:
        print(f"\n--- 通过主键查询 ids={ids} ---")

        results_by_id = milvus_client.query(
            collection_name=COLLECTION_NAME,
            ids=ids[:1],  # 只查询第一个
            output_fields=["id", "text", "source"],
        )

        for item in results_by_id:
            print(
                f"id: {item['id']}, "
                f"text: {item['text']}, "
                f"source: {item['source']}"
            )

    # ========= 删除操作 =========
    print("\n" + "=" * 50)
    print("2. 删除：删除实体")
    print("=" * 50)

    # 通过主键删除
    if ids:
        print(f"\n--- 通过主键删除 ids={ids[:1]} ---")

        milvus_client.delete(
            collection_name=COLLECTION_NAME,
            ids=ids[:1],
        )

        print("Deleted!")

    # 通过过滤表达式删除
    print("\n--- 通过过滤表达式删除 source == 'language' ---")

    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter='source == "language"',
    )

    print("Deleted!")

    # 验证删除结果
    print("\n--- 验证：查询剩余数据 ---")

    results = milvus_client.query(
        collection_name=COLLECTION_NAME,
        output_fields=["id", "text", "source"],
    )

    print(f"Remaining: {len(results)} documents")

    for item in results:
        print(f"  id: {item['id']}, text: {item['text']}")

    # ========= 加载操作 =========
    print("\n" + "=" * 50)
    print("3. 加载：重新加载 Collection")
    print("=" * 50)

    # 卸载 Collection
    milvus_client.release_collection(COLLECTION_NAME)
    print(f"Released collection '{COLLECTION_NAME}'")

    # 重新加载
    milvus_client.load_collection(COLLECTION_NAME)
    print(f"Loaded collection '{COLLECTION_NAME}'")

    # ========= 删除 Collection =========
    print("\n" + "=" * 50)
    print("4. 删除 Collection：删除整个集合")
    print("=" * 50)

    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"Dropped collection '{COLLECTION_NAME}'")

    # 验证
    exists = milvus_client.has_collection(COLLECTION_NAME)
    print(f"Collection exists: {exists}")

    print("\n" + "=" * 50)
    print("All operations completed!")
    print("=" * 50)


if __name__ == "__main__":
    if not API_KEY:
        print("Error: Please set SILICONFLOW_API_KEY")
        exit(1)

    main()