# graph/graph_rag.py
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from .neo4j_connector import Neo4jConnector
import numpy as np
from sentence_transformers import SentenceTransformer

class GraphRAG:
    """基于知识图谱的检索增强生成"""
    
    def __init__(self, neo4j_connector: Neo4jConnector):
        self.neo4j = neo4j_connector
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # 构建内存中的图结构用于快速遍历
        self.graph = nx.MultiDiGraph()
        self._load_graph_structure()
    
    def _load_graph_structure(self):
        """从Neo4j加载图结构到NetworkX"""
        query = """
        MATCH (n)-[r]->(m)
        RETURN labels(n) as source_labels, n.name as source_name,
               type(r) as relationship_type,
               labels(m) as target_labels, m.name as target_name,
               properties(r) as rel_properties
        """
        
        with self.neo4j.driver.session() as session:
            result = session.run(query)
            for record in result:
                source = f"{record['source_labels'][0]}:{record['source_name']}"
                target = f"{record['target_labels'][0]}:{record['target_name']}"
                
                self.graph.add_edge(
                    source, target,
                    relationship=record['relationship_type'],
                    **record['rel_properties']
                )
    
    def semantic_search_with_graph(self, query: str, max_depth: int = 2, top_k: int = 10) -> List[Dict[str, Any]]:
        """结合语义搜索和图遍历的检索"""
        
        # 1. 首先进行语义匹配找到起始节点
        initial_nodes = self._semantic_node_matching(query, top_k=5)
        
        # 2. 从起始节点进行图遍历
        expanded_subgraphs = []
        for node_info in initial_nodes:
            subgraph = self._expand_subgraph(node_info['node_id'], max_depth)
            subgraph['initial_similarity'] = node_info['similarity']
            expanded_subgraphs.append(subgraph)
        
        # 3. 对子图进行排序和筛选
        ranked_subgraphs = self._rank_subgraphs(query, expanded_subgraphs)
        
        # 4. 生成结构化结果
        results = []
        for subgraph in ranked_subgraphs[:top_k]:
            result = self._subgraph_to_context(subgraph)
            results.append(result)
        
        return results
    
    def _semantic_node_matching(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """语义匹配图中的节点"""
        # 获取所有节点的文本描述
        node_texts = []
        node_ids = []
        
        for node in self.graph.nodes():
            node_type, node_name = node.split(':', 1)
            # 构建节点的文本描述
            node_text = f"{node_type}: {node_name}"
            
            # 添加连接信息增强语义
            neighbors = list(self.graph.successors(node))
            if neighbors:
                neighbor_info = [n.split(':', 1)[1] for n in neighbors[:3]]
                node_text += f" 相关: {', '.join(neighbor_info)}"
            
            node_texts.append(node_text)
            node_ids.append(node)
        
        # 计算语义相似度
        if not node_texts:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        node_embeddings = self.embedding_model.encode(node_texts)
        
        # 计算余弦相似度
        similarities = np.dot(query_embedding, node_embeddings.T).flatten()
        
        # 排序并返回top_k
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            results.append({
                'node_id': node_ids[idx],
                'node_text': node_texts[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def _expand_subgraph(self, start_node: str, max_depth: int) -> Dict[str, Any]:
        """从起始节点扩展子图"""
        subgraph_nodes = {start_node}
        subgraph_edges = []
        
        current_level = {start_node}
        
        for depth in range(max_depth):
            next_level = set()
            
            for node in current_level:
                # 获取出边
                for successor in self.graph.successors(node):
                    if successor not in subgraph_nodes:
                        next_level.add(successor)
                        subgraph_nodes.add(successor)
                    
                    # 记录边信息
                    for edge_data in self.graph[node][successor].values():
                        subgraph_edges.append({
                            'source': node,
                            'target': successor,
                            'relationship': edge_data.get('relationship', ''),
                            'properties': {k: v for k, v in edge_data.items() if k != 'relationship'}
                        })
                
                # 获取入边
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in subgraph_nodes:
                        next_level.add(predecessor)
                        subgraph_nodes.add(predecessor)
                    
                    for edge_data in self.graph[predecessor][node].values():
                        subgraph_edges.append({
                            'source': predecessor,
                            'target': node,
                            'relationship': edge_data.get('relationship', ''),
                            'properties': {k: v for k, v in edge_data.items() if k != 'relationship'}
                        })
            
            current_level = next_level
            
            # 限制子图大小
            if len(subgraph_nodes) > 50:
                break
        
        return {
            'start_node': start_node,
            'nodes': list(subgraph_nodes),
            'edges': subgraph_edges,
            'depth': depth + 1
        }
    
    def _rank_subgraphs(self, query: str, subgraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对子图进行排序"""
        for subgraph in subgraphs:
            # 计算子图的综合得分
            score = 0.0
            
            # 初始相似度权重
            score += subgraph.get('initial_similarity', 0) * 0.4
            
            # 子图大小奖励（但不能太大）
            node_count = len(subgraph['nodes'])
            size_score = min(node_count / 20.0, 1.0) * 0.2
            score += size_score
            
            # 连接密度奖励
            edge_count = len(subgraph['edges'])
            if node_count > 1:
                density = edge_count / (node_count * (node_count - 1))
                score += density * 0.2
            
            # 语义相关性
            subgraph_text = self._subgraph_to_text(subgraph)
            if subgraph_text:
                query_emb = self.embedding_model.encode([query])
                subgraph_emb = self.embedding_model.encode([subgraph_text])
                semantic_sim = np.dot(query_emb, subgraph_emb.T)[0, 0]
                score += semantic_sim * 0.2
            
            subgraph['final_score'] = score
        
        # 按得分排序
        return sorted(subgraphs, key=lambda x: x['final_score'], reverse=True)
    
    def _subgraph_to_text(self, subgraph: Dict[str, Any]) -> str:
        """将子图转换为文本描述"""
        text_parts = []
        
        # 添加节点信息
        nodes_by_type = {}
        for node in subgraph['nodes']:
            node_type, node_name = node.split(':', 1)
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node_name)
        
        for node_type, names in nodes_by_type.items():
            text_parts.append(f"{node_type}: {', '.join(names[:5])}")
        
        # 添加关系信息
        relationships = set()
        for edge in subgraph['edges']:
            rel = edge['relationship']
            if rel:
                relationships.add(rel)
        
        if relationships:
            text_parts.append(f"关系: {', '.join(relationships)}")
        
        return "; ".join(text_parts)
    
    def _subgraph_to_context(self, subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """将子图转换为上下文信息"""
        # 组织节点信息
        entities = {}
        for node in subgraph['nodes']:
            node_type, node_name = node.split(':', 1)
            if node_type not in entities:
                entities[node_type] = []
            entities[node_type].append(node_name)
        
        # 组织关系信息
        relationships = []
        for edge in subgraph['edges']:
            relationships.append({
                'source': edge['source'].split(':', 1)[1],
                'target': edge['target'].split(':', 1)[1],
                'relationship': edge['relationship'],
                'properties': edge['properties']
            })
        
        return {
            'subgraph_id': subgraph['start_node'],
            'entities': entities,
            'relationships': relationships,
            'context_text': self._subgraph_to_text(subgraph),
            'relevance_score': subgraph.get('final_score', 0),
            'depth': subgraph['depth']
        }
    
    def get_treatment_pathway(self, disease: str, patient_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """获取治疗路径"""
        # 从图中查找治疗路径
        pathways = self.neo4j.find_treatment_paths(disease, patient_profile)
        
        # 构建路径图
        pathway_graph = {
            'disease': disease,
            'pathways': [],
            'decision_points': []
        }
        
        for pathway in pathways:
            pathway_info = {
                'treatment': pathway['treatment'],
                'type': pathway['treatment_type'],
                'efficacy': pathway.get('efficacy', 0),
                'evidence_level': pathway.get('evidence_level', ''),
                'line_of_therapy': pathway.get('line_of_therapy', 1),
                'drugs': pathway.get('drugs', [])
            }
            pathway_graph['pathways'].append(pathway_info)
        
        return pathway_graph
    
    def analyze_case_with_graph(self, case_symptoms: List[str], case_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """使用图谱分析病例"""
        
        # 1. 找到相似病例
        similar_cases = self.neo4j.find_similar_cases(case_symptoms)
        
        # 2. 为每个症状进行图搜索
        symptom_analyses = []
        for symptom in case_symptoms:
            analysis = self.semantic_search_with_graph(f"症状 {symptom}", max_depth=2, top_k=3)
            symptom_analyses.append({
                'symptom': symptom,
                'related_info': analysis
            })
        
        # 3. 综合分析
        analysis_result = {
            'input_symptoms': case_symptoms,
            'similar_cases': similar_cases,
            'symptom_analyses': symptom_analyses,
            'recommendations': self._generate_graph_recommendations(similar_cases, symptom_analyses)
        }
        
        return analysis_result
    
    def _generate_graph_recommendations(self, similar_cases: List[Dict[str, Any]], 
                                      symptom_analyses: List[Dict[str, Any]]) -> List[str]:
        """基于图分析生成推荐"""
        recommendations = []
        
        # 基于相似病例的推荐
        if similar_cases:
            top_disease = similar_cases[0]
            recommendations.append(f"考虑诊断: {top_disease['disease']} (症状匹配度: {top_disease['symptom_overlap']})")
            
            # 推荐治疗
            treatments = top_disease.get('treatments', [])
            if treatments:
                recommendations.append(f"推荐治疗方案: {', '.join(treatments[:3])}")
        
        # 基于症状分析的推荐
        all_entities = {}
        for analysis in symptom_analyses:
            for info in analysis['related_info']:
                entities = info.get('entities', {})
                for entity_type, entity_list in entities.items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = set()
                    all_entities[entity_type].update(entity_list)
        
        # 推荐进一步检查
        if 'Treatment' in all_entities:
            treatments = list(all_entities['Treatment'])[:3]
            recommendations.append(f"可考虑的治疗选项: {', '.join(treatments)}")
        
        if 'Drug' in all_entities:
            drugs = list(all_entities['Drug'])[:3]
            recommendations.append(f"相关药物: {', '.join(drugs)}")
        
        return recommendations if recommendations else ["需要更多信息进行分析"]