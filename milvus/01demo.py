# 示例：将图像转换为向量
# 假设使用预训练模型提取图像特征
import numpy as np
# 一张224x224的RGB图像，经过CNN模型提取后
# 可能得到一个2048维的向量
image_embedding = np.random.randn(2048).astype(np.float32)
print(f"图像向量维度: {image_embedding.shape}")
print(f"向量前5个值: {image_embedding[:5]}")
# 一段文本，经过BERT模型处理后
# 可能得到一个768维的向量
text_embedding = np.random.randn(768).astype(np.float32)
print(f"文本向量维度: {text_embedding.shape}")