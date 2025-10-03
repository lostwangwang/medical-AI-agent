import json
import time
from typing import List, Dict, Any
from pathlib import Path
from agents.base_agent import MedicalCase, AgentRole
from agents.oncologist import OncologistAgent
from agents.radiologist import RadiologistAgent
from agents.nurse import NurseAgent
from agents.psychologist import PsychologistAgent
from agents.patient_advocate import PatientAdvocateAgent
from consensus.decision_engine import DecisionEngine
from rag.knowledge_base import MedicalKnowledgeBase
from .evaluator import MedicalAIEvaluator

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config_file: str = "experiment_config.json"):
        self.config = self._load_config(config_file)
        
        # 初始化系统组件
        self.knowledge_base = MedicalKnowledgeBase()
        self.decision_engine = DecisionEngine()
        self.evaluator = MedicalAIEvaluator()
        
        # 初始化智能体
        self.agents = {
            AgentRole.ONCOLOGIST: OncologistAgent(),
            AgentRole.RADIOLOGIST: RadiologistAgent(),
            AgentRole.NURSE: NurseAgent(),
            AgentRole.PSYCHOLOGIST: PsychologistAgent(),
            AgentRole.PATIENT_ADVOCATE: PatientAdvocateAgent()
        }
        
        # 更新智能体的知识库
        for agent in self.agents.values():
            agent.update_knowledge_base(self.knowledge_base)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载实验配置"""
        default_config = {
            "test_dataset_path": "data/test_cases.json",
            "clinical_guidelines_path": "data/clinical_guidelines.json",
            "output_dir": "experiment_results",
            "batch_size": 10,
            "include_agents": ["oncologist", "radiologist", "nurse", "psychologist", "patient_advocate"],
            "evaluation_metrics": ["diagnosis_accuracy", "treatment_appropriateness", "consensus_quality"],
            "comparison_baselines": ["clinical_guidelines", "expert_opinions"]
        }
        
        if Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def load_test_dataset(self) -> List[Dict[str, Any]]:
        """加载测试数据集"""
        test_cases = []
        
        with open(self.config["test_dataset_path"], 'r', encoding='utf-8') as f:
            raw_cases = json.load(f)
        
        for raw_case in raw_cases:
            # 转换为MedicalCase对象
            case = MedicalCase(
                case_id=raw_case['case_id'],
                patient_info=raw_case['patient_info'],
                symptoms=raw_case['symptoms'],
                medical_history=raw_case['medical_history'],
                test_results=raw_case['test_results'],
                imaging_data=raw_case['imaging_data']
            )
            
            test_cases.append({
                'case': case,
                'ground_truth': raw_case.get('ground_truth', {}),
                'expected_outcomes': raw_case.get('expected_outcomes', {})
            })
        
        return test_cases
    
    def run_single_case_experiment(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个案例实验"""
        
        case = test_case['case']
        ground_truth = test_case['ground_truth']
        
        start_time = time.time()
        
        # 收集各智能体的意见
        agent_opinions = []
        
        for role, agent in self.agents.items():
            if role.value in self.config["include_agents"]:
                try:
                    opinion = agent.analyze_case(case)
                    agent_opinions.append(opinion)
                except Exception as e:
                    print(f"Error in {role.value} analysis: {e}")
                    continue
        
        # 生成决策
        decision_result = self.decision_engine.make_decision(case, agent_opinions)
        
        analysis_time = time.time() - start_time
        
        # 评估结果
        evaluation_result = self.evaluator.evaluate_case_analysis(
            case, agent_opinions, ground_truth
        )
        
        return {
            'case_id': case.case_id,
            'agent_opinions': agent_opinions,
            'decision_result': decision_result,
            'evaluation_result': evaluation_result,
            'analysis_time': analysis_time,
            'ground_truth': ground_truth
        }
    
    def run_batch_experiment(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行批量实验"""
        
        print(f"Starting batch experiment with {len(test_cases)} cases...")
        
        experiment_results = {
            'experiment_config': self.config,
            'start_time': time.time(),
            'case_results': [],
            'batch_evaluation': {},
            'performance_metrics': {}
        }
        
        # 按批次处理
        batch_size = self.config.get("batch_size", 10)
        
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(test_cases)-1)//batch_size + 1}")
            
            batch_results = []
            
            for test_case in batch:
                try:
                    case_result = self.run_single_case_experiment(test_case)
                    batch_results.append(case_result)
                    experiment_results['case_results'].append(case_result)
                    
                    print(f"  Completed case {case_result['case_id']}: "
                          f"Score {case_result['evaluation_result']['overall_score']:.3f}")
                    
                except Exception as e:
                    print(f"  Error processing case {test_case['case']['case_id']}: {e}")
                    continue
        
        experiment_results['end_time'] = time.time()
        experiment_results['total_duration'] = experiment_results['end_time'] - experiment_results['start_time']
        
        # 批量评估
        experiment_results['batch_evaluation'] = self._evaluate_batch_performance(
            experiment_results['case_results']
        )
        
        # 性能指标
        experiment_results['performance_metrics'] = self._calculate_performance_metrics(
            experiment_results['case_results']
        )
        
        return experiment_results
    
    def _evaluate_batch_performance(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估批量性能"""
        
        # 准备数据用于评估器
        test_cases_for_eval = []
        for result in case_results:
            test_cases_for_eval.append({
                'case': type('MockCase', (), {'case_id': result['case_id']})(),
                'agent_opinions': result['agent_opinions'],
                'ground_truth': result['ground_truth']
            })
        
        return self.evaluator.evaluate_batch(test_cases_for_eval, self.decision_engine)
    
    def _calculate_performance_metrics(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算性能指标"""
        
        if not case_results:
            return {}
        
        analysis_times = [r['analysis_time'] for r in case_results]
        overall_scores = [r['evaluation_result']['overall_score'] for r in case_results]
        
        metrics = {
            'average_analysis_time': sum(analysis_times) / len(analysis_times),
            'median_analysis_time': sorted(analysis_times)[len(analysis_times)//2],
            'max_analysis_time': max(analysis_times),
            'min_analysis_time': min(analysis_times),
            
            'average_score': sum(overall_scores) / len(overall_scores),
            'score_std': (sum((s - sum(overall_scores)/len(overall_scores))**2 for s in overall_scores) / len(overall_scores))**0.5,
            'high_score_cases': len([s for s in overall_scores if s >= 0.8]),
            'low_score_cases': len([s for s in overall_scores if s < 0.6]),
            
            'throughput': len(case_results) / sum(analysis_times) * 3600,  # cases per hour
        }
        
        return metrics
    
    def compare_with_baselines(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """与基线方法对比"""
        
        comparison_results = {
            'baseline_comparisons': {},
            'improvement_analysis': {},
            'statistical_significance': {}
        }
        
        # 与临床指南对比
        if "clinical_guidelines" in self.config["comparison_baselines"]:
            try:
                with open(self.config["clinical_guidelines_path"], 'r', encoding='utf-8') as f:
                    guidelines = json.load(f)
                
                guideline_comparison = self.evaluator.compare_with_guidelines(
                    experiment_results['batch_evaluation'], guidelines
                )
                
                comparison_results['baseline_comparisons']['clinical_guidelines'] = guideline_comparison
                
            except FileNotFoundError:
                print("Clinical guidelines file not found, skipping comparison")
        
        # 可以添加更多基线对比...
        
        return comparison_results
    
    def save_experiment_results(self, experiment_results: Dict[str, Any], 
                               output_dir: str = None) -> None:
        """保存实验结果"""
        
        if output_dir is None:
            output_dir = self.config["output_dir"]
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        results_file = Path(output_dir) / f"experiment_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 需要处理不可序列化的对象
            serializable_results = self._make_serializable(experiment_results)
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 生成评估报告
        report_file = Path(output_dir) / f"evaluation_report_{timestamp}.json"
        self.evaluator.generate_evaluation_report(
            experiment_results['batch_evaluation'], 
            str(report_file)
        )
        
        print(f"Experiment results saved to {results_file}")
        print(f"Evaluation report saved to {report_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化格式"""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # for Enum objects
            return obj.value
        else:
            return obj
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """运行完整实验"""
        
        print("=== 医疗多智能体系统实验开始 ===")
        
        # 1. 加载知识库
        print("1. 加载医疗知识库...")
        self.knowledge_base.load_clinical_guidelines("data/guidelines/")
        self.knowledge_base.load_research_papers("data/papers/")
        self.knowledge_base.load_drug_information("data/drugs/")
        
        # 2. 加载测试数据
        print("2. 加载测试数据集...")
        test_cases = self.load_test_dataset()
        print(f"   加载了 {len(test_cases)} 个测试案例")
        
        # 3. 运行批量实验
        print("3. 运行批量实验...")
        experiment_results = self.run_batch_experiment(test_cases)
        
        # 4. 基线对比
        print("4. 与基线方法对比...")
        comparison_results = self.compare_with_baselines(experiment_results)
        experiment_results['baseline_comparisons'] = comparison_results
        
        # 5. 保存结果
        print("5. 保存实验结果...")
        self.save_experiment_results(experiment_results)
        
        # 6. 打印总结
        self._print_experiment_summary(experiment_results)
        
        print("=== 实验完成 ===")
        
        return experiment_results
    
    def _print_experiment_summary(self, results: Dict[str, Any]) -> None:
        """打印实验总结"""
        
        print("\n=== 实验总结 ===")
        print(f"总测试案例数: {len(results['case_results'])}")
        print(f"总耗时: {results['total_duration']:.2f} 秒")
        
        if 'batch_evaluation' in results and 'avg_scores' in results['batch_evaluation']:
            avg_scores = results['batch_evaluation']['avg_scores']
            print(f"平均整体得分: {avg_scores['overall']:.3f}")
            print(f"平均诊断得分: {avg_scores['diagnosis']:.3f}")
            print(f"平均治疗得分: {avg_scores['treatment']:.3f}")
            print(f"平均共识得分: {avg_scores['consensus']:.3f}")
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"平均分析时间: {metrics['average_analysis_time']:.2f} 秒")
            print(f"系统吞吐量: {metrics['throughput']:.1f} 案例/小时")
            print(f"高分案例数: {metrics['high_score_cases']}")
            print(f"低分案例数: {metrics['low_score_cases']}")