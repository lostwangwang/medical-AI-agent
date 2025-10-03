# consensus/decision_engine.py
from typing import List, Dict, Any
from agents.base_agent import AgentOpinion, MedicalCase, AgentRole
from consensus.consensus_matrix import ConsensusMatrix, ConsensusResult


class DecisionEngine:
    def __init__(self):
        self.consensus_matrix = ConsensusMatrix()
        self.decision_history = []
    
    def make_decision(self, case: MedicalCase, opinions: List[AgentOpinion]) -> Dict[str, Any]:
        """基于多智能体意见做出决策"""
        
        # 1. 计算共识
        consensus_result = self.consensus_matrix.calculate_consensus(opinions)
        
        # 2. 生成决策报告
        decision_report = {
            'case_id': case.case_id,
            'consensus_result': consensus_result,
            'agent_opinions': opinions,
            'decision_summary': self._generate_decision_summary(consensus_result),
            'next_steps': self._determine_next_steps(consensus_result, case),
            'follow_up_plan': self._create_follow_up_plan(consensus_result),
            'quality_metrics': self._calculate_quality_metrics(opinions, consensus_result)
        }
        
        # 3. 记录决策历史
        self.decision_history.append(decision_report)
        
        return decision_report

    def _generate_decision_summary(self, consensus_result: ConsensusResult) -> str:
        """生成决策摘要"""
        summary = f"""
        决策摘要：
        
        共识度：{consensus_result.consensus_score:.2f} (满分1.0)
        优先级：{consensus_result.weighted_priority:.1f} (满分10.0)
        置信区间：{consensus_result.confidence_interval[0]:.2f} - {consensus_result.confidence_interval[1]:.2f}
        
        推荐方案：
        {consensus_result.final_recommendation}
        
        风险评估：
        """

        for risk_type, risk_level in consensus_result.risk_assessment.items():
            risk_desc = "低" if risk_level < 0.3 else "中" if risk_level < 0.7 else "高"
            summary += f"- {risk_type}: {risk_desc} ({risk_level:.2f})\n"

        if consensus_result.dissenting_opinions:
            summary += "\n需要注意的不同意见：\n"
            for dissent in consensus_result.dissenting_opinions:
                summary += f"- {dissent}\n"

        return summary

    def _determine_next_steps(self, consensus_result: ConsensusResult, case: MedicalCase) -> List[str]:
        """确定下一步行动"""
        next_steps = []
        
        # 基于共识度决定行动
        if consensus_result.consensus_score < 0.5:
            next_steps.append("需要进一步的多学科讨论")
            next_steps.append("考虑获取第二诊疗意见")
        
        # 基于风险评估决定行动
        risks = consensus_result.risk_assessment
        
        if risks.get('medical_risk', 0) > 0.7:
            next_steps.append("紧急医疗评估和监护")
        
        if risks.get('psychological_risk', 0) > 0.5:
            next_steps.append("心理健康专业干预")
        
        if risks.get('economic_risk', 0) > 0.5:
            next_steps.append("社会工作者介入，评估经济支持需求")
        
        # 基于优先级决定紧急度
        if consensus_result.weighted_priority > 8.0:
            next_steps.insert(0, "高优先级处理，48小时内启动治疗")
        elif consensus_result.weighted_priority > 6.0:
            next_steps.insert(0, "中等优先级，一周内完成相关检查")
        
        return next_steps if next_steps else ["按计划执行推荐方案"]
    
    def _create_follow_up_plan(self, consensus_result: ConsensusResult) -> Dict[str, Any]:
        """创建随访计划"""
        follow_up = {
            'short_term': [],  # 1-2周
            'medium_term': [], # 1-3月
            'long_term': []    # 3月以上
        }
        
        # 基于风险水平制定随访计划
        risks = consensus_result.risk_assessment
        
        # 短期随访
        if any(risk > 0.6 for risk in risks.values()):
            follow_up['short_term'].extend([
                "症状监测和评估",
                "治疗反应评价",
                "副作用管理"
            ])
        
        # 中期随访
        follow_up['medium_term'].extend([
            "疗效评估",
            "生活质量评估",
            "心理状态评估"
        ])
        
        # 长期随访
        follow_up['long_term'].extend([
            "定期复查",
            "复发监测",
            "康复评估"
        ])
        
        return follow_up
    
    def _calculate_quality_metrics(self, opinions: List[AgentOpinion], consensus_result: ConsensusResult) -> Dict[str, float]:
        """计算决策质量指标"""
        return {
            'opinion_diversity': len(set(op.role for op in opinions)) / len(AgentRole),
            'average_confidence': sum(op.confidence for op in opinions) / len(opinions),
            'consensus_strength': consensus_result.consensus_score,
            'risk_coverage': len([r for r in consensus_result.risk_assessment.values() if r > 0]) / len(consensus_result.risk_assessment),
            'recommendation_completeness': min(len(consensus_result.final_recommendation.split('\n')) / 10, 1.0)
        }