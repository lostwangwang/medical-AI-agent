# rag/retriever.py
from typing import List, Dict, Any, Optional
from .knowledge_base import MedicalKnowledgeBase
from agents.base_agent import MedicalCase

class MedicalRAGRetriever:
    """医疗RAG检索器"""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        
        # 不同查询类型的权重配置
        self.query_weights = {
            'diagnosis': {
                'clinical_guidelines': 0.4,
                'research_papers': 0.3,
                'case_studies': 0.3
            },
            'treatment': {
                'clinical_guidelines': 0.5,
                'drug_information': 0.3,
                'research_papers': 0.2
            },
            'drug_info': {
                'drug_information': 0.6,
                'clinical_guidelines': 0.3,
                'research_papers': 0.1
            }
        }
    
    def retrieve_for_case(self, case: MedicalCase, query_type: str = 'diagnosis', 
                         top_k: int = 10) -> List[Dict[str, Any]]:
        """为特定病例检索相关知识"""
        
        # 构建查询字符串
        query_components = []
        
        # 添加症状信息
        if case.symptoms:
            query_components.append(f"症状: {', '.join(case.symptoms)}")
        
        # 添加诊断信息
        if case.test_results and 'suspected_diagnosis' in case.test_results:
            query_components.append(f"诊断: {case.test_results['suspected_diagnosis']}")
        
        # 添加患者信息
        if case.patient_info:
            age = case.patient_info.get('age', '')
            gender = case.patient_info.get('gender', '')
            if age or gender:
                query_components.append(f"患者: {age}岁 {gender}")
        
        # 组合查询
        base_query = " ".join(query_components)
        
        # 根据查询类型调整搜索策略
        all_results = []
        weights = self.query_weights.get(query_type, self.query_weights['diagnosis'])
        
        for category, weight in weights.items():
            if weight > 0:
                # 每个类别的搜索结果数量按权重分配
                category_top_k = max(1, int(top_k * weight))
                category_results = self.knowledge_base.search_knowledge(
                    base_query, category=category, top_k=category_top_k
                )
                
                # 添加权重信息
                for result in category_results:
                    result['category_weight'] = weight
                    result['weighted_score'] = result.get('similarity_score', 0) * weight
                
                all_results.extend(category_results)
        
        # 按加权分数排序
        all_results.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        
        # 返回前top_k个结果
        return all_results[:top_k]
    
    def retrieve_for_question(self, question: str, context: Optional[Dict[str, Any]] = None,
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """为特定问题检索相关知识"""
        
        # 分析问题类型
        query_type = self._analyze_question_type(question)
        
        # 增强查询
        enhanced_query = question
        if context:
            # 添加上下文信息
            if 'case_info' in context:
                enhanced_query += f" {context['case_info']}"
        
        # 搜索知识库
        results = []
        weights = self.query_weights.get(query_type, self.query_weights['diagnosis'])
        
        for category, weight in weights.items():
            if weight > 0:
                category_top_k = max(1, int(top_k * weight))
                category_results = self.knowledge_base.search_knowledge(
                    enhanced_query, category=category, top_k=category_top_k
                )
                
                for result in category_results:
                    result['category_weight'] = weight
                    result['weighted_score'] = result.get('similarity_score', 0) * weight
                
                results.extend(category_results)
        
        # 排序并返回
        results.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        return results[:top_k]
    
    def _analyze_question_type(self, question: str) -> str:
        """分析问题类型"""
        question_lower = question.lower()
        
        # 治疗相关关键词
        treatment_keywords = ['治疗', '用药', '方案', '化疗', '放疗', '手术', '靶向', '免疫']
        if any(keyword in question_lower for keyword in treatment_keywords):
            return 'treatment'
        
        # 药物相关关键词
        drug_keywords = ['药物', '副作用', '剂量', '禁忌', '相互作用']
        if any(keyword in question_lower for keyword in drug_keywords):
            return 'drug_info'
        
        # 默认为诊断类型
        return 'diagnosis'
    
    def get_context_for_agent(self, agent_role: str, case: MedicalCase) -> str:
        """为特定角色的智能体获取上下文信息"""
        
        # 根据角色调整检索策略
        role_specific_queries = {
            'oncologist': f"肿瘤科 {case.test_results.get('suspected_diagnosis', '')} 治疗指南",
            'radiologist': f"影像学诊断 {' '.join(case.symptoms)} CT MRI",
            'nurse': f"护理 {' '.join(case.symptoms)} 症状管理 患者教育",
            'psychologist': f"肿瘤心理 焦虑抑郁 心理支持 生活质量",
            'patient_advocate': f"患者权益 知情同意 经济负担 生活质量"
        }
        
        query = role_specific_queries.get(agent_role, "")
        if not query:
            return ""
        
        # 检索相关信息
        results = self.knowledge_base.search_knowledge(query, top_k=3)
        
        # 构建上下文
        context_parts = []
        for result in results:
            title = result.get('title', '')
            content = result.get('content', '')[:500]  # 限制长度
            context_parts.append(f"【{title}】\n{content}...")
        
        return "\n\n".join(context_parts)