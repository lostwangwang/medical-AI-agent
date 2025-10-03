import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from agents.base_agent import AgentOpinion, AgentRole
import json

@dataclass
class ConsensusResult:
    final_recommendation: str
    consensus_score: float  # 0-1，越高代表共识度越高
    weighted_priority: float  # 加权优先级评分
    dissenting_opinions: List[str]  # 不同意见
    confidence_interval: Tuple[float, float]  # 置信区间
    risk_assessment: Dict[str, float]  # 风险评估

class ConsensusMatrix:
    def __init__(self):
        # 角色权重配置
        self.role_weights = {
            AgentRole.ONCOLOGIST: 0.35,      # 肿瘤科医生权重最高
            AgentRole.RADIOLOGIST: 0.25,     # 影像科医生
            AgentRole.NURSE: 0.15,           # 护士
            AgentRole.PSYCHOLOGIST: 0.15,    # 心理咨询师
            AgentRole.PATIENT_ADVOCATE: 0.10  # 患者代表
        }
        
        # 决策维度权重
        self.decision_dimensions = {
            'medical_accuracy': 0.40,    # 医疗准确性
            'patient_safety': 0.30,     # 患者安全
            'quality_of_life': 0.20,    # 生活质量
            'cost_effectiveness': 0.10  # 成本效益
        }
    
    def calculate_consensus(self, opinions: List[AgentOpinion]) -> ConsensusResult:
        """计算多智能体共识"""
        if not opinions:
            raise ValueError("至少需要一个意见进行共识计算")
        
        # 1. 提取关键指标
        confidences = [op.confidence for op in opinions]
        priorities = [op.priority_score for op in opinions]
        role_weights = [self.role_weights.get(op.role, 0.2) for op in opinions]
        
        # 2. 计算加权置信度
        weighted_confidence = np.average(confidences, weights=role_weights)
        
        # 3. 计算加权优先级
        weighted_priority = np.average(priorities, weights=role_weights)
        
        # 4. 计算共识度 - 基于意见一致性
        consensus_score = self._calculate_consensus_score(opinions)
        
        # 5. 生成最终推荐
        final_recommendation = self._generate_final_recommendation(opinions, role_weights)
        
        # 6. 识别不同意见
        dissenting_opinions = self._identify_dissenting_opinions(opinions, consensus_score)
        
        # 7. 计算置信区间
        confidence_interval = self._calculate_confidence_interval(confidences, role_weights)
        
        # 8. 风险评估
        risk_assessment = self._assess_risks(opinions)
        
        return ConsensusResult(
            final_recommendation=final_recommendation,
            consensus_score=consensus_score,
            weighted_priority=weighted_priority,
            dissenting_opinions=dissenting_opinions,
            confidence_interval=confidence_interval,
            risk_assessment=risk_assessment
        )
    
    def _calculate_consensus_score(self, opinions: List[AgentOpinion]) -> float:
        """计算共识度分数"""
        if len(opinions) <= 1:
            return 1.0
        
        # 基于推荐建议的重叠度计算共识
        all_recommendations = []
        for op in opinions:
            all_recommendations.extend(op.recommendations)
        
        # 计算建议重叠度
        unique_recommendations = set(all_recommendations)
        total_recommendations = len(all_recommendations)
        
        if total_recommendations == 0:
            return 0.0
        
        # 计算每个建议被提及的频率
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # 共识度 = 重叠建议的比例
        overlapping_count = sum(count for count in recommendation_counts.values() if count > 1)
        consensus_score = overlapping_count / total_recommendations
        
        # 考虑置信度的一致性
        confidence_std = np.std([op.confidence for op in opinions])
        confidence_consistency = 1 - min(confidence_std, 0.5) / 0.5
        
        # 综合共识度
        final_consensus = 0.7 * consensus_score + 0.3 * confidence_consistency
        return min(final_consensus, 1.0)
    
    def _generate_final_recommendation(self, opinions: List[AgentOpinion], role_weights: List[float]) -> str:
        """生成最终推荐方案"""
        # 收集所有推荐建议
        weighted_recommendations = {}
        
        for i, opinion in enumerate(opinions):
            weight = role_weights[i] * opinion.confidence
            for rec in opinion.recommendations:
                if rec in weighted_recommendations:
                    weighted_recommendations[rec] += weight
                else:
                    weighted_recommendations[rec] = weight
        
        # 按权重排序
        sorted_recommendations = sorted(
            weighted_recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 构建最终推荐
        final_rec = "综合多学科意见，推荐治疗方案：\n\n"
        
        # 高优先级推荐 (权重 > 平均权重)
        avg_weight = np.mean(list(weighted_recommendations.values()))
        high_priority = [rec for rec, weight in sorted_recommendations if weight > avg_weight]
        
        if high_priority:
            final_rec += "核心推荐：\n"
            for i, rec in enumerate(high_priority[:5], 1):  # 最多5个核心推荐
                final_rec += f"{i}. {rec}\n"
        
        # 补充建议
        other_recommendations = [rec for rec, weight in sorted_recommendations if weight <= avg_weight]
        if other_recommendations:
            final_rec += "\n补充建议：\n"
            for i, rec in enumerate(other_recommendations[:3], 1):  # 最多3个补充建议
                final_rec += f"{i}. {rec}\n"
        
        # 添加重要关注点
        all_concerns = []
        for opinion in opinions:
            all_concerns.extend(opinion.concerns)
        
        if all_concerns:
            final_rec += "\n重要关注点：\n"
            unique_concerns = list(set(all_concerns))[:3]  # 最多3个关注点
            for i, concern in enumerate(unique_concerns, 1):
                final_rec += f"{i}. {concern}\n"
        
        return final_rec
    
    def _identify_dissenting_opinions(self, opinions: List[AgentOpinion], consensus_score: float) -> List[str]:
        """识别不同意见"""
        if consensus_score > 0.8:  # 高共识度
            return []
        
        dissenting = []
        
        # 找出明显不同的意见
        for opinion in opinions:
            if opinion.confidence < 0.5:  # 低置信度意见
                dissenting.append(f"{opinion.role.value}: 置信度较低({opinion.confidence:.2f}) - {opinion.opinion[:100]}...")
            
            # 检查是否有独特的关注点
            other_concerns = []
            for other_op in opinions:
                if other_op.agent_id != opinion.agent_id:
                    other_concerns.extend(other_op.concerns)
            
            unique_concerns = set(opinion.concerns) - set(other_concerns)
            if unique_concerns:
                dissenting.append(f"{opinion.role.value}: 独特关注 - {list(unique_concerns)}")
        
        return dissenting
    
    def _calculate_confidence_interval(self, confidences: List[float], weights: List[float]) -> Tuple[float, float]:
        """计算置信区间"""
        weighted_mean = np.average(confidences, weights=weights)
        weighted_std = np.sqrt(np.average((np.array(confidences) - weighted_mean)**2, weights=weights))
        
        # 95% 置信区间
        margin_error = 1.96 * weighted_std / np.sqrt(len(confidences))
        lower_bound = max(0, weighted_mean - margin_error)
        upper_bound = min(1, weighted_mean + margin_error)
        
        return (lower_bound, upper_bound)
    
    def _assess_risks(self, opinions: List[AgentOpinion]) -> Dict[str, float]:
        """评估各种风险"""
        risks = {
            'medical_risk': 0.0,      # 医疗风险
            'psychological_risk': 0.0,  # 心理风险
            'economic_risk': 0.0,     # 经济风险
            'quality_of_life_risk': 0.0  # 生活质量风险
        }
        
        # 基于各角色的关注点评估风险
        for opinion in opinions:
            if opinion.role == AgentRole.ONCOLOGIST:
                # 医疗风险主要来自肿瘤科医生的评估
                risk_keywords = ['转移', '复发', '副作用', '并发症']
                medical_risk = sum(1 for concern in opinion.concerns 
                                 if any(keyword in concern for keyword in risk_keywords))
                risks['medical_risk'] = min(medical_risk / len(opinion.concerns) if opinion.concerns else 0, 1.0)
            
            elif opinion.role == AgentRole.PSYCHOLOGIST:
                # 心理风险评估
                psych_risk_keywords = ['抑郁', '焦虑', '自杀', '创伤']
                psych_risk = sum(1 for concern in opinion.concerns 
                               if any(keyword in concern for keyword in psych_risk_keywords))
                risks['psychological_risk'] = min(psych_risk / len(opinion.concerns) if opinion.concerns else 0, 1.0)
            
            elif opinion.role == AgentRole.PATIENT_ADVOCATE:
                # 经济和生活质量风险
                economic_keywords = ['费用', '经济', '负担']
                quality_keywords = ['生活质量', '功能', '自理']
                
                economic_risk = sum(1 for concern in opinion.concerns 
                                  if any(keyword in concern for keyword in economic_keywords))
                quality_risk = sum(1 for concern in opinion.concerns 
                                 if any(keyword in concern for keyword in quality_keywords))
                
                risks['economic_risk'] = min(economic_risk / len(opinion.concerns) if opinion.concerns else 0, 1.0)
                risks['quality_of_life_risk'] = min(quality_risk / len(opinion.concerns) if opinion.concerns else 0, 1.0)
        
        return risks