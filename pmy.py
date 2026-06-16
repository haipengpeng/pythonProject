# -*- coding: utf-8 -*-
import random
import base64
import json
import requests

# 生成随机logid
logid = random.randint(1000000, 100000000)

# 读取图片文件
imgfile = 'porn.jpg'
img_data = open(imgfile, 'rb').read()

# 创建请求对象
req_obj = {
    'image': base64.b64encode(img_data).decode('utf-8'),
    'type_name': 'disgust',
    'enable_feature': 0
}

# 将请求对象转换为JSON格式并进行编码
data = json.dumps(req_obj).encode('utf-8')

# 创建请求数组
req_array = {
    'appid': '123456',
    'logid': logid,
    'format': 'json',
    'from': 'test-python',
    'cmdid': '123',
    'clientip': '0.0.0.0',
    'data': base64.b64encode(data).decode('utf-8')
}

# 将请求数组转换为JSON格式
req_json = json.dumps(req_array)

# 发送POST请求
response = requests.post(url='http://localhost:8233/GeneralClassifyService/classify', data=req_json, headers={'Content-Type': 'application/json'})

# 打印响应文本
#print(response.text)

# 解析响应头
print("Response Headers:")
print(response.headers)

# 解析响应体
print("\nResponse Body:")
print(response.text)

# 假设返回的JSON字符串存储在response_text变量中
response_text = response.text

# 解析JSON响应
response_dict = json.loads(response_text)

# 提取并解码result字段
encoded_result = response_dict['result']
decoded_result = base64.b64decode(encoded_result).decode('utf-8')

# 解析解码后的result字段
result_dict = json.loads(decoded_result)

# 打印解析后的result字段
print("\nParsed Result:")
print(json.dumps(result_dict, indent=4, ensure_ascii=False))
