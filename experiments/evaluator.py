import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime
from agents.base_agent import AgentOpinion, MedicalCase
from consensus.decision_engine import DecisionEngine

class MedicalAIEvaluator:
    """医疗AI系统评估器"""
    
    def __init__(self):
        self.evaluation_metrics = {
            'diagnosis_accuracy': [],
            'treatment_appropriateness': [],
            'consensus_quality': [],
            'response_time': [],
            'user_satisfaction': []
        }
        
        # 评估标准
        self.evaluation_criteria = {
            'diagnosis': {
                'accuracy': 'Correct diagnosis identification',
                'completeness': 'Comprehensive differential diagnosis',
                'confidence': 'Appropriate confidence levels'
            },
            'treatment': {
                'guideline_adherence': 'Adherence to clinical guidelines',
                'personalization': 'Patient-specific considerations',
                'safety': 'Safety profile assessment'
            },
            'consensus': {
                'agreement_level': 'Level of inter-agent agreement',
                'rationale_quality': 'Quality of reasoning',
                'decision_consistency': 'Consistency with evidence'
            }
        }
    
    def evaluate_case_analysis(self, case: MedicalCase, agent_opinions: List[AgentOpinion], 
                             ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """评估单个病例的分析质量"""
        
        results = {
            'overall_score': 0.0,
            'diagnosis_score': 0.0,
            'treatment_score': 0.0,
            'consensus_score': 0.0,
            'detail_scores': {}
        }
        
        # 1. 诊断准确性评估
        diagnosis_score = self._evaluate_diagnosis(agent_opinions, ground_truth.get('true_diagnosis'))
        results['diagnosis_score'] = diagnosis_score
        
        # 2. 治疗方案评估
        treatment_score = self._evaluate_treatment(agent_opinions, ground_truth.get('recommended_treatment'))
        results['treatment_score'] = treatment_score
        
        # 3. 共识质量评估
        consensus_score = self._evaluate_consensus_quality(agent_opinions)
        results['consensus_score'] = consensus_score
        
        # 4. 计算总分
        results['overall_score'] = (diagnosis_score * 0.4 + treatment_score * 0.4 + consensus_score * 0.2)
        
        # 5. 详细评分
        results['detail_scores'] = {
            'agent_confidence_avg': np.mean([op.confidence for op in agent_opinions]),
            'agent_confidence_std': np.std([op.confidence for op in agent_opinions]),
            'recommendation_overlap': self._calculate_recommendation_overlap(agent_opinions),
            'concern_coverage': self._evaluate_concern_coverage(agent_opinions, ground_truth)
        }
        
        return results
    
    def _evaluate_diagnosis(self, opinions: List[AgentOpinion], true_diagnosis: str) -> float:
        """评估诊断准确性"""
        if not true_diagnosis:
            return 0.0
        
        diagnosis_mentions = 0
        total_diagnostic_opinions = 0
        
        for opinion in opinions:
            # 只考虑医生的诊断意见
            if opinion.role.value in ['oncologist', 'radiologist']:
                total_diagnostic_opinions += 1
                
                # 检查是否提到了正确的诊断
                if true_diagnosis.lower() in opinion.opinion.lower():
                    diagnosis_mentions += 1
        
        if total_diagnostic_opinions == 0:
                return 0.0
        
        return diagnosis_mentions / total_diagnostic_opinions
    
    def _evaluate_treatment(self, opinions: List[AgentOpinion], recommended_treatment: List[str]) -> float:
        """评估治疗方案的合理性"""
        if not recommended_treatment:
            return 0.0
        
        treatment_score = 0.0
        total_treatment_opinions = 0
        
        for opinion in opinions:
            if opinion.role.value in ['oncologist', 'nurse']:
                total_treatment_opinions += 1
                
                # 检查推荐的治疗方案
                opinion_recommendations = set([rec.lower() for rec in opinion.recommendations])
                ground_truth_treatments = set([treat.lower() for treat in recommended_treatment])
                
                # 计算交集比例
                overlap = len(opinion_recommendations.intersection(ground_truth_treatments))
                if len(ground_truth_treatments) > 0:
                    treatment_score += overlap / len(ground_truth_treatments)
        
        return treatment_score / total_treatment_opinions if total_treatment_opinions > 0 else 0.0
    
    def _evaluate_consensus_quality(self, opinions: List[AgentOpinion]) -> float:
        """评估共识质量"""
        if len(opinions) <= 1:
            return 1.0
        
        # 1. 置信度一致性
        confidences = [op.confidence for op in opinions]
        confidence_consistency = 1 - (np.std(confidences) / np.mean(confidences)) if np.mean(confidences) > 0 else 0
        
        # 2. 推荐建议重叠度
        recommendation_overlap = self._calculate_recommendation_overlap(opinions)
        
        # 3. 关注点互补性
        all_concerns = []
        for op in opinions:
            all_concerns.extend(op.concerns)
        
        unique_concerns = len(set(all_concerns))
        total_concerns = len(all_concerns)
        concern_diversity = unique_concerns / total_concerns if total_concerns > 0 else 0
        
        # 综合评分
        consensus_quality = (confidence_consistency * 0.4 + 
                           recommendation_overlap * 0.4 + 
                           concern_diversity * 0.2)
        
        return min(consensus_quality, 1.0)
    
    def _calculate_recommendation_overlap(self, opinions: List[AgentOpinion]) -> float:
        """计算推荐建议的重叠度"""
        all_recommendations = []
        for op in opinions:
            all_recommendations.extend([rec.lower() for rec in op.recommendations])
        
        if not all_recommendations:
            return 0.0
        
        # 计算重复推荐的比例
        from collections import Counter
        rec_counts = Counter(all_recommendations)
        overlapping_recs = sum(count for count in rec_counts.values() if count > 1)
        
        return overlapping_recs / len(all_recommendations)
    
    def _evaluate_concern_coverage(self, opinions: List[AgentOpinion], ground_truth: Dict[str, Any]) -> float:
        """评估关注点覆盖度"""
        expected_concerns = ground_truth.get('expected_concerns', [])
        if not expected_concerns:
            return 1.0
        
        all_concerns = []
        for op in opinions:
            all_concerns.extend([concern.lower() for concern in op.concerns])
        
        covered_concerns = 0
        for expected in expected_concerns:
            if any(expected.lower() in concern for concern in all_concerns):
                covered_concerns += 1
        
        return covered_concerns / len(expected_concerns)
    
    def evaluate_batch(self, test_cases: List[Dict[str, Any]], 
                      decision_engine: DecisionEngine) -> Dict[str, Any]:
        """批量评估测试用例"""
        
        batch_results = {
            'total_cases': len(test_cases),
            'avg_scores': {},
            'score_distribution': {},
            'case_results': [],
            'summary_metrics': {}
        }
        
        all_scores = []
        diagnosis_scores = []
        treatment_scores = []
        consensus_scores = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Evaluating case {i+1}/{len(test_cases)}: {test_case['case']['case_id']}")
            
            case = test_case['case']
            opinions = test_case['agent_opinions']
            ground_truth = test_case['ground_truth']
            
            # 评估单个案例
            case_result = self.evaluate_case_analysis(case, opinions, ground_truth)
            case_result['case_id'] = case.case_id
            
            batch_results['case_results'].append(case_result)
            
            # 收集分数用于统计
            all_scores.append(case_result['overall_score'])
            diagnosis_scores.append(case_result['diagnosis_score'])
            treatment_scores.append(case_result['treatment_score'])
            consensus_scores.append(case_result['consensus_score'])
        
        # 计算平均分和分布
        batch_results['avg_scores'] = {
            'overall': np.mean(all_scores),
            'diagnosis': np.mean(diagnosis_scores),
            'treatment': np.mean(treatment_scores),
            'consensus': np.mean(consensus_scores)
        }
        
        batch_results['score_distribution'] = {
            'overall_std': np.std(all_scores),
            'high_performers': len([s for s in all_scores if s >= 0.8]),
            'low_performers': len([s for s in all_scores if s < 0.6])
        }
        
        # 生成总结指标
        batch_results['summary_metrics'] = self._generate_summary_metrics(batch_results)
        
        return batch_results
    
    def _generate_summary_metrics(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成总结性指标"""
        case_results = batch_results['case_results']
        
        metrics = {
            'system_reliability': batch_results['score_distribution']['high_performers'] / batch_results['total_cases'],
            'areas_for_improvement': [],
            'strength_areas': [],
            'recommendations': []
        }
        
        # 识别需要改进的领域
        avg_scores = batch_results['avg_scores']
        if avg_scores['diagnosis'] < 0.7:
            metrics['areas_for_improvement'].append('诊断准确性需要提升')
        if avg_scores['treatment'] < 0.7:
            metrics['areas_for_improvement'].append('治疗方案需要优化')
        if avg_scores['consensus'] < 0.7:
            metrics['areas_for_improvement'].append('多智能体共识机制需要改进')
        
        # 识别优势领域
        if avg_scores['diagnosis'] >= 0.8:
            metrics['strength_areas'].append('诊断能力表现优秀')
        if avg_scores['treatment'] >= 0.8:
            metrics['strength_areas'].append('治疗方案制定能力强')
        if avg_scores['consensus'] >= 0.8:
            metrics['strength_areas'].append('多智能体协作效果好')
        
        # 生成改进建议
        if len(metrics['areas_for_improvement']) > 0:
            metrics['recommendations'].append('需要增加相关领域的训练数据')
            metrics['recommendations'].append('考虑调整模型参数或架构')
        
        return metrics
    
    def compare_with_guidelines(self, test_results: Dict[str, Any], 
                               clinical_guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """与临床指南对比评估"""
        
        comparison_results = {
            'guideline_adherence': 0.0,
            'deviation_analysis': [],
            'improvement_suggestions': []
        }
        
        adherence_scores = []
        
        for case_result in test_results['case_results']:
            case_id = case_result['case_id']
            
            # 检查是否有对应的指南
            if case_id in clinical_guidelines:
                guideline = clinical_guidelines[case_id]
                
                # 计算指南遵循度
                adherence = self._calculate_guideline_adherence(case_result, guideline)
                adherence_scores.append(adherence)
                
                if adherence < 0.8:
                    comparison_results['deviation_analysis'].append({
                        'case_id': case_id,
                        'adherence_score': adherence,
                        'deviations': self._identify_deviations(case_result, guideline)
                    })
        
        comparison_results['guideline_adherence'] = np.mean(adherence_scores) if adherence_scores else 0.0
        
        return comparison_results
    
    def _calculate_guideline_adherence(self, case_result: Dict[str, Any], 
                                     guideline: Dict[str, Any]) -> float:
        """计算指南遵循度"""
        # 这里需要根据具体的指南格式来实现
        # 简化版本：比较推荐的治疗方案
        
        adherence_score = 0.0
        total_criteria = 0
        
        # 检查诊断标准
        if 'diagnosis_criteria' in guideline:
            total_criteria += 1
            if case_result['diagnosis_score'] >= 0.8:
                adherence_score += 1
        
        # 检查治疗标准
        if 'treatment_criteria' in guideline:
            total_criteria += 1
            if case_result['treatment_score'] >= 0.8:
                adherence_score += 1
        
        return adherence_score / total_criteria if total_criteria > 0 else 0.0
    
    def _identify_deviations(self, case_result: Dict[str, Any], 
                           guideline: Dict[str, Any]) -> List[str]:
        """识别与指南的偏差"""
        deviations = []
        
        if case_result['diagnosis_score'] < 0.8:
            deviations.append('诊断准确性低于指南标准')
        
        if case_result['treatment_score'] < 0.8:
            deviations.append('治疗方案不符合指南推荐')
        
        return deviations
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                 output_file: str) -> None:
        """生成评估报告"""
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'summary': evaluation_results['summary_metrics'],
            'detailed_results': evaluation_results,
            'recommendations': self._generate_improvement_recommendations(evaluation_results)
        }
        
        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成可读性报告
        readable_report = self._format_readable_report(report)
        readable_file = output_file.replace('.json', '_readable.txt')
        
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(readable_report)
        
        print(f"Evaluation report saved to {output_file}")
        print(f"Readable report saved to {readable_file}")
    
    def _generate_improvement_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        avg_scores = results['avg_scores']
        
        if avg_scores['overall'] < 0.7:
            recommendations.append("整体性能需要提升，建议增加训练数据量和优化模型架构")
        
        if avg_scores['diagnosis'] < avg_scores['treatment']:
            recommendations.append("诊断能力相对较弱，建议加强医学知识图谱和诊断推理训练")
        
        if avg_scores['consensus'] < 0.6:
            recommendations.append("多智能体协作效果不佳，建议优化共识机制和角色定义")
        
        if results['score_distribution']['low_performers'] > results['total_cases'] * 0.2:
            recommendations.append("存在较多低性能案例，建议分析失败案例并针对性改进")
        
        return recommendations
    
    def _format_readable_report(self, report: Dict[str, Any]) -> str:
        """格式化可读性报告"""
        
        readable_text = f"""
医疗多智能体系统评估报告
=====================================

评估时间: {report['evaluation_timestamp']}

总体性能概览
-----------
- 系统可靠性: {report['summary']['system_reliability']:.1%}
- 总测试案例数: {report['detailed_results']['total_cases']}
- 高性能案例: {report['detailed_results']['score_distribution']['high_performers']}
- 低性能案例: {report['detailed_results']['score_distribution']['low_performers']}

平均得分
--------
- 整体得分: {report['detailed_results']['avg_scores']['overall']:.3f}
- 诊断准确性: {report['detailed_results']['avg_scores']['diagnosis']:.3f}
- 治疗方案: {report['detailed_results']['avg_scores']['treatment']:.3f}
- 共识质量: {report['detailed_results']['avg_scores']['consensus']:.3f}

优势领域
--------
"""
        
        for strength in report['summary']['strength_areas']:
            readable_text += f"✓ {strength}\n"
        
        readable_text += "\n需要改进的领域\n------------\n"
        
        for improvement in report['summary']['areas_for_improvement']:
            readable_text += f"⚠ {improvement}\n"
        
        readable_text += "\n改进建议\n--------\n"
        
        for i, rec in enumerate(report['recommendations'], 1):
            readable_text += f"{i}. {rec}\n"
        
        return readable_text