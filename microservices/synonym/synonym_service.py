import fasttext
import numpy as np
import time
from functools import lru_cache
from loguru import logger
from ms_core import load_config, ModelConfig


class FastTextSynonymExpander:
    def __init__(self):
        """
        初始化同义词扩展器
        
        Args:
            model_path: FastText模型路径
            top_n_words: 加载的总词数
            preload_top: 预加载的高频词数量
        """
        try:
            # 从 nacos 获取 synonym 服务配置
            config = load_config("model_config")
            if not isinstance(config, ModelConfig):
                raise ValueError
            model_path = config.synonym.model_path or None # 同义词模型路径
            top_n_words = config.synonym.top_n_words or 50000 # 加载的总词数
            preload_top = config.synonym.preload_top or 1000 # 预加载的高频词数量
        except Exception as e:
            # 如果从 nacos 获取 synonym 服务配置失败，则使用默认配置
            logger.warning("Failed to get synonym service config from nacos: {}".format(e))
            model_path = None
            top_n_words = 50000
            preload_top = 1000
        
        if model_path is None:
            logger.exception("FastText模型路径未配置")
            raise ValueError("FastText模型路径未配置")
        
        self.model_path = model_path
        self.top_n_words = top_n_words
        self.preload_top = preload_top
        self._is_initialized = False

    
    async def load_model(self):
        """加载FastText模型和词向量"""
        logger.info(f"开始加载FastText模型: {self.model_path}")
        start_time = time.time()
        
        # 加载模型
        self.model = fasttext.load_model(self.model_path)
        
        # 获取词汇表（按频率排序）
        words, freqs = self.model.get_words(include_freq=True)
        word_freq_pairs = sorted(zip(words, freqs), key=lambda x: -x[1])
        
        # 仅保留高频词
        self.words = [w for w, f in word_freq_pairs[:self.top_n_words]]
        self.word_index = {w: i for i, w in enumerate(self.words)}
        
        # 预加载所有词向量到连续内存
        self.vectors = np.zeros((len(self.words), 300), dtype=np.float32)
        for i, word in enumerate(self.words):
            self.vectors[i] = self.model.get_word_vector(word)
        
        # 预计算归一化向量（加速余弦相似度）
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.norm_vectors = self.vectors / np.where(norms > 0, norms, 1)
        
        logger.info(f"模型加载完成，共加载 {len(self.words)} 个词，耗时: {time.time()-start_time:.2f}s")
        self._is_initialized = True
    
    @lru_cache(maxsize=100000)
    def get_synonym(self, word: str, top_k: int | None = 2, threshold: float | None = 0.6) -> list[str]:
        """
        获取同义词（带LRU缓存）
        
        Args:
            word: 查询词
            top_k: 返回的同义词数量
            threshold: 相似度阈值
            
        Returns:
            同义词列表
        """
        if word not in self.word_index:
            return []
        
        idx = self.word_index[word]
        target_vec = self.norm_vectors[idx]

        if top_k is None:
            top_k = 2
        if threshold is None:
            threshold = 0.6
        
        # 计算相似度（向量化操作）
        similarities = np.dot(self.norm_vectors, target_vec)
        
        # 获取最相似的前topk+1个词（排除自己）
        top_indices = np.argpartition(-similarities, top_k+1)[:top_k+1]
        top_indices = top_indices[top_indices != idx][:top_k]  # 排除自身
        
        # 过滤结果
        return [self.words[i] for i in top_indices if similarities[i] >= threshold]
    
    async def preload_cache(self):
        """预加载高频词的同义词到缓存"""
        if self.preload_top <= 0:
            return
            
        logger.info(f"开始预加载前{self.preload_top}个高频词的同义词缓存...")
        start_time = time.time()
        
        # 使用批量查询提高效率
        batch_size = 100  # 每批处理的词数
        words_to_preload = self.words[:self.preload_top]
        
        for i in range(0, len(words_to_preload), batch_size):
            batch = words_to_preload[i:i+batch_size]
            for word in batch:
                # 调用被lru_cache装饰的方法会自动缓存结果
                self.get_synonym(word)
        
        logger.info(f"预加载完成，共加载{len(words_to_preload)}个词，耗时: {time.time()-start_time:.2f}s")
    
    async def get_cache_info(self) -> dict:
        """获取缓存统计信息"""
        cache_info = self.get_synonym.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) 
                        if (cache_info.hits + cache_info.misses) > 0 else 0
        }
    
    async def shutdown(self):
        """关闭并清理资源"""
        if not self._is_initialized:
            return
            
        logger.info("开始关闭同义词扩展器并清理资源...")
        start_time = time.time()
        
        # 清理FastText模型
        if hasattr(self, 'model'):
            del self.model
            logger.info("已释放FastText模型")
        
        # 清理缓存
        if hasattr(self, 'get_synonyms'):
            self.get_synonym.cache_clear()
            logger.info("已清空同义词缓存")
        
        # 清理其他大对象
        for attr in ['words', 'word_index', 'vectors', 'norm_vectors']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self._is_initialized = False
        logger.info(f"资源清理完成，耗时: {time.time()-start_time:.2f}s")