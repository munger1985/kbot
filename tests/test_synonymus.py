import fasttext
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from functools import lru_cache
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedFastTextSynonymExpander:
    def __init__(self, model_path: str, top_n_words: int = 50000):
        """
        高度优化的FastText同义词扩展器
        
        Args:
            model_path: FastText模型路径
            top_n_words: 仅加载前N个高频词（平衡性能与内存）
        """
        start_time = time.time()
        
        # 加载模型
        self.model = fasttext.load_model(model_path)
        
        # 获取词汇表（按频率排序）
        words, freqs = self.model.get_words(include_freq=True)
        word_freq_pairs = sorted(zip(words, freqs), key=lambda x: -x[1])
        
        # 仅保留高频词
        self.words = [w for w, f in word_freq_pairs[:top_n_words]]
        self.word_index = {w: i for i, w in enumerate(self.words)}
        
        # 预加载所有词向量到连续内存
        self.vectors = np.zeros((len(self.words), 300), dtype=np.float32)
        for i, word in enumerate(self.words):
            self.vectors[i] = self.model.get_word_vector(word)
        
        # 预计算归一化向量（加速余弦相似度）
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.norm_vectors = self.vectors / np.where(norms > 0, norms, 1)
        
        # 构建快速查询索引
        self._build_index()
        
        logger.info(f"初始化完成，加载 {len(self.words)} 个词，耗时: {time.time()-start_time:.2f}s")

    def _build_index(self):
        """构建快速查询索引"""
        # 这里可以添加更高级的索引结构，如Annoy或FAISS
        # 当前使用简单的内存数组
        pass
    @lru_cache(maxsize=10000)
    def get_synonyms(self, word: str, topk: int = 3, threshold: float = 0.6) -> List[str]:
        """
        获取同义词（优化版）
        
        Args:
            word: 查询词
            topk: 返回的同义词数量
            threshold: 相似度阈值
            
        Returns:
            同义词列表
        """
        if word not in self.word_index:
            return []
        
        idx = self.word_index[word]
        target_vec = self.norm_vectors[idx]
        
        # 计算相似度（向量化操作）
        similarities = np.dot(self.norm_vectors, target_vec)
        
        # 获取最相似的前topk+1个词（排除自己）
        top_indices = np.argpartition(-similarities, topk+1)[:topk+1]
        top_indices = top_indices[top_indices != idx][:topk]  # 排除自身
        
        # 过滤结果
        results = []
        for i in top_indices:
            if similarities[i] >= threshold:
                results.append(self.words[i])
        
        return results

    def batch_get_synonyms(self, words: List[str], topk: int = 3) -> Dict[str, List[str]]:
        """
        批量获取同义词（更高效）
        
        Args:
            words: 查询词列表
            topk: 每个词返回的同义词数量
            
        Returns:
            字典格式的同义词结果
        """
        # 找出所有存在的词
        valid_words = [w for w in words if w in self.word_index]
        if not valid_words:
            return {}
        
        # 获取所有向量
        indices = [self.word_index[w] for w in valid_words]
        target_vecs = self.norm_vectors[indices]
        
        # 批量计算相似度
        similarities = np.dot(self.norm_vectors, target_vecs.T)
        
        # 处理每个查询词
        results = {}
        for i, word in enumerate(valid_words):
            # 获取最相似的前topk+1个词（排除自己）
            top_indices = np.argpartition(-similarities[:, i], topk+1)[:topk+1]
            top_indices = top_indices[top_indices != indices[i]][:topk]
            
            # 收集结果
            results[word] = [self.words[j] for j in top_indices 
                            if similarities[j, i] >= 0.6]
        
        return results

# 使用示例
if __name__ == "__main__":
    # 初始化（只需一次）
    logger.info("正在初始化同义词扩展器...")
    expander = OptimizedFastTextSynonymExpander(
        model_path="/home/chris/models/cc.zh.300.bin",
        top_n_words=50000  # 根据内存调整
    )
    
    # 示例查询
    test_words = ["电脑", "病毒", "程序", "学习"]
    
    # 单次查询
    for word in test_words:
        start = time.time()
        synonyms = expander.get_synonyms(word)
        logger.info(f"'{word}' 的同义词: {synonyms} (耗时: {(time.time()-start)*1000:.2f}ms)")
    
    # 批量查询（更高效）
    start = time.time()
    batch_results = expander.batch_get_synonyms(test_words)
    logger.info(f"批量查询结果: {batch_results} (总耗时: {(time.time()-start)*1000:.2f}ms)")