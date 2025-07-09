#!/usr/bin/env python3
"""
嵌入服务客户端示例

此脚本展示如何使用嵌入服务的API进行常见操作：
- 生成文本嵌入
- 更新模型配置
- 检查服务健康状态
- 获取服务统计信息
"""

import requests
import json
import time
import numpy as np
from typing import List, Dict, Any

class EmbeddingServiceClient:
    """嵌入服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: 服务基础URL
        """
        self.base_url = base_url.rstrip('/')
    
    def embed(self, texts: List[str], model_id: str = "text2vec") -> np.ndarray:
        """
        生成文本嵌入
        
        Args:
            texts: 要嵌入的文本列表
            model_id: 要使用的模型ID
            
        Returns:
            numpy.ndarray: 嵌入向量数组
        """
        url = f"{self.base_url}/api/embedding/embed"
        payload = {
            "texts": texts,
            "model_id": model_id
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return np.array(result["embeddings"])
    
    def update_model_config(self, model_id: str, config: Dict[str, Any], version: str = None) -> bool:
        """
        更新模型配置
        
        Args:
            model_id: 模型ID
            config: 新的配置
            version: 配置版本（可选）
            
        Returns:
            bool: 是否更新成功
        """
        url = f"{self.base_url}/api/embedding/models/{model_id}/config"
        payload = {
            "config": config
        }
        if version:
            payload["version"] = version
        
        response = requests.put(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["updated"]
    
    def check_health(self) -> Dict[str, Any]:
        """
        检查服务健康状态
        
        Returns:
            Dict: 健康状态信息
        """
        url = f"{self.base_url}/api/embedding/health"
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取服务统计信息
        
        Returns:
            Dict: 统计信息
        """
        url = f"{self.base_url}/api/embedding/stats"
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()

def main():
    """主函数"""
    # 创建客户端
    client = EmbeddingServiceClient("http://localhost:8000")
    
    try:
        # 检查服务健康状态
        print("检查服务健康状态...")
        health = client.check_health()
        print(f"服务状态: {health['status']}")
        print(f"时间戳: {health['timestamp']}")
        print("模型状态:")
        for model_id, status in health['models'].items():
            print(f"  - {model_id}: {status['status']}")
        print()
        
        # 获取服务统计信息
        print("获取服务统计信息...")
        stats = client.get_stats()
        print(f"实例ID: {stats['instance_id']}")
        print(f"CPU使用率: {stats['cpu_usage']:.2f}%")
        print(f"内存使用率: {stats['memory_usage']:.2f}%")
        print("模型统计:")
        for model_id, model_stats in stats['models'].items():
            print(f"  - {model_id}: 请求次数={model_stats['request_count']}, 最后使用时间={time.ctime(model_stats['last_used'])}")
        print()
        
        # 更新模型配置
        print("更新模型配置...")
        new_config = {
            "api_url": "https://api.example.com/embedding",
            "api_key": "demo_key_123",
            "dimensions": 768,
            "max_tokens": 512
        }
        updated = client.update_model_config("text2vec", new_config)
        print(f"配置更新{'成功' if updated else '失败'}")
        print()
        
        # 生成文本嵌入
        print("生成文本嵌入...")
        texts = [
            "这是第一个测试文本",
            "这是第二个测试文本，内容稍微长一些",
            "这是第三个测试文本，包含一些特殊字符：!@#$%^&*()"
        ]
        embeddings = client.embed(texts, "text2vec")
        print(f"生成了{len(embeddings)}个嵌入向量")
        print(f"向量维度: {embeddings.shape[1]}")
        print(f"第一个向量的前10个元素: {embeddings[0][:10]}")
        
        # 计算相似度示例
        print("\n计算文本相似度示例:")
        # 归一化向量
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 计算余弦相似度
        similarity = np.dot(normalized, normalized.T)
        print("相似度矩阵:")
        print(similarity)
        
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()