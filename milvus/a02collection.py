from pymilvus import MilvusClient
# 连接到你 docker-compose 里的 Milvus
milvus_client = MilvusClient(uri="http://localhost:19530")  # 这里的 uri 写法是官方示例推荐的
collection_name = "rag_docs"
embedding_dim = 1536  # 根据你的 embedding 模型实际维度调整
# 如果 collection 已存在，可以先删掉重建（生产环境慎用 drop）
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
# 创建 collection（自动 schema，dynamic field 会自动帮你存 text / source 等非预定义字段）
milvus_client.create_collection(
collection_name=collection_name,
    dimension=embedding_dim,
    auto_id=True,                    # 自动分配id 相当于MySQL中auto_increment
    metric_type="IP",                # 与 OpenAI RAG 示例一致，使用内积
    consistency_level="Bounded",     # 一般 RAG 场景用 Bounded / Strong 均可
)
print("Collection created:", collection_name)
# 这一段代码会：
# 自动创建：
# id（主键，int64）
# vector（FLOAT_VECTOR，dim=1536）
# 保留一个 dynamic field，称为元数据（JSON），你可以随意插入 text / source / created_at 等字段