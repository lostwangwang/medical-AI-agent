import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
import os

import faiss
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.exceptions import AuthenticationFailedException


class VectorStore:
    """向量存储基类"""
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量"""
        return self.embedding_model.encode(texts)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档"""
        raise NotImplementedError

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        raise NotImplementedError

class FAISSVectorStore(VectorStore):
    """基于FAISS的向量存储"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "medical_knowledge.faiss"):
        super().__init__(embedding_model)
        self.index_path = index_path
        self.metadata_path = index_path.replace('.faiss', '_metadata.pkl')

        # 初始化FAISS索引
        self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
        self.metadata = []

        # 如果存在已保存的索引，则加载
        if os.path.exists(index_path):
            self.load_index()

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档到向量存储"""
        if not documents:
            return

        # 提取文本内容
        texts = []
        for doc in documents:
            # 组合文档的各个字段
            content = ""
            if 'title' in doc:
                content += f"标题: {doc['title']}\n"
            if 'content' in doc:
                content += f"内容: {doc['content']}\n"
            if 'summary' in doc:
                content += f"摘要: {doc['summary']}\n"
            texts.append(content.strip())

        # 编码为向量
        embeddings = self.encode_texts(texts)

        # 添加到FAISS索引
        self.index.add(embeddings)

        # 保存元数据
        self.metadata.extend(documents)

        print(f"Added {len(documents)} documents to vector store")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        if self.index.ntotal == 0:
            return []

        # 编码查询
        query_vector = self.encode_texts([query])

        # 搜索
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        # 返回结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # 有效索引
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)

        return results

    def save_index(self) -> None:
        """保存索引到磁盘"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved to {self.index_path}")

    def load_index(self) -> None:
        """从磁盘加载索引"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Index loaded from {self.index_path}")

class WeaviateVectorStore(VectorStore):
    """基于Weaviate的向量存储"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 weaviate_url: str = "http://localhost:8080"):
        super().__init__(embedding_model)

        try:
            self.client = weaviate.WeaviateClient(weaviate_url)
            self.class_name = "MedicalDocument"
            self._setup_schema()
            print("Weaviate 连接成功")
        except (ConnectionError, AuthenticationFailedException) as e:
            print(f"Weaviate connection failed: {e}")
            self.client = None
        except Exception as e:
            print(f"未知错误: {e}")
            self.client = None

    def _setup_schema(self):
        """设置Weaviate模式"""
        schema = {
            "class": self.class_name,
            "description": "Medical knowledge documents",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Document title"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document content"
                },
                {
                    "name": "category",
                    "dataType": ["string"],
                    "description": "Document category"
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                    "description": "Document source"
                }
            ],
            "vectorizer": "none"  # 使用自定义向量
        }

        # 检查类是否存在
        if not self.client.schema.exists(self.class_name):
            self.client.schema.create_class(schema)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档到Weaviate"""
        if not self.client or not documents:
            return

        # 批量添加
        with self.client.batch as batch:
            for doc in documents:
                # 提取文本内容用于向量化
                content = ""
                if 'title' in doc:
                    content += f"{doc['title']} "
                if 'content' in doc:
                    content += doc['content']

                # 生成向量
                vector = self.encode_texts([content])[0]

                # 准备数据对象
                data_object = {
                    "title": doc.get('title', ''),
                    "content": doc.get('content', ''),
                    "category": doc.get('category', ''),
                    "source": doc.get('source', '')
                }

                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.class_name,
                    vector=vector.tolist()
                )

        print(f"Added {len(documents)} documents to Weaviate")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """在Weaviate中搜索"""
        if not self.client:
            return []

        # 生成查询向量
        query_vector = self.encode_texts([query])[0]

        # 执行向量搜索
        result = (
            self.client.query
            .get(self.class_name, ["title", "content", "category", "source"])
            .with_near_vector({
                "vector": query_vector.tolist()
            })
            .with_limit(top_k)
            .with_additional(["certainty"])
            .do()
        )

        # 处理结果
        documents = []
        if "data" in result and "Get" in result["data"]:
            docs = result["data"]["Get"][self.class_name]
            for i, doc in enumerate(docs):
                doc['similarity_score'] = doc['_additional']['certainty']
                doc['rank'] = i + 1
                del doc['_additional']
                documents.append(doc)

        return documents