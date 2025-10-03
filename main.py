import argparse
import json
from pathlib import Path
import os
from dotenv import load_dotenv

import time

from experiments.experiment_runner import ExperimentRunner
from agents.base_agent import MedicalCase
from consensus.decision_engine import DecisionEngine
from rag.knowledge_base import MedicalKnowledgeBase

def main():
    """主程序入口"""
    
    parser = argparse.ArgumentParser(description="医疗多智能体角色扮演系统")
    parser.add_argument("--mode", choices=["demo", "experiment", "interactive", "neo4j"], 
                       default="demo", help="运行模式")
    parser.add_argument("--config", default="experiment_config.json", 
                       help="配置文件路径")
    parser.add_argument("--case-file", help="单个案例文件路径")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo_case()
    elif args.mode == "experiment":
        run_full_experiment(args.config)
    elif args.mode == "interactive":
        run_interactive_mode()
    elif args.mode == "neo4j":
        run_neo4j_interactive_mode()

def run_demo_case():
    """运行演示案例"""
    
    print("=== 医疗多智能体系统演示 ===")
    
    # 创建演示案例
    demo_case = MedicalCase(
        case_id="demo_001",
        patient_info={
            "age": 45,
            "gender": "女性",
            "weight": 65,
            "height": 165
        },
        symptoms=[
            "右乳肿块",
            "轻微疼痛",
            "肿块增大"
        ],
        medical_history=[
            "无恶性肿瘤家族史",
            "月经规律",
            "未服用激素类药物"
        ],
        test_results={
            "血常规": "正常",
            "肝肾功能": "正常",
            "肿瘤标志物": "CA153略高于正常",
            "suspected_diagnosis": "乳腺恶性肿瘤"
        },
        imaging_data={
            "钼靶": "右乳外上象限4cm不规则肿块，边缘毛刺",
            "超声": "低回声肿块，血流信号丰富",
            "MRI": "T1WI低信号，T2WI稍高信号，强化明显"
        }
    )
    
    # 初始化系统
    from agents.oncologist import OncologistAgent
    from agents.radiologist import RadiologistAgent
    from agents.nurse import NurseAgent
    from agents.psychologist import PsychologistAgent
    from agents.patient_advocate import PatientAdvocateAgent
    
    # 创建智能体
    agents = [
        OncologistAgent(),
        RadiologistAgent(),
        NurseAgent(),
        PsychologistAgent(),
        PatientAdvocateAgent()
    ]
    
    # 收集意见
    print("\n收集各角色意见...")
    opinions = []
    for agent in agents:
        print(f"\n{agent.role.value}分析中...")
        opinion = agent.analyze_case(demo_case)
        print(f"{agent.role.value}的建议:", opinion)
        opinions.append(opinion)
        
        print(f"置信度: {opinion.confidence:.2f}")
        print(f"优先级: {opinion.priority_score:.1f}")
        print(f"主要建议: {opinion.recommendations[:2]}")
    
    # 生成决策
    print("\n生成多学科决策...")
    decision_engine = DecisionEngine()
    decision = decision_engine.make_decision(demo_case, opinions)
    
    print("\n=== 最终决策报告 ===")
    print(decision['decision_summary'])
    
    print("\n=== 后续行动计划 ===")
    for step in decision['next_steps']:
        print(f"• {step}")

def run_full_experiment(config_file: str):
    """运行完整实验"""
    
    # 检查配置文件
    if not Path(config_file).exists():
        print(f"配置文件 {config_file} 不存在，创建默认配置...")
        create_default_config(config_file)
    
    # 运行实验
    runner = ExperimentRunner(config_file)
    results = runner.run_full_experiment()
    
    return results

def run_neo4j_interactive_mode():
    """从Neo4j数据库获取数据并提供命令行交互"""
    
    print("=== 医疗知识图谱 - Neo4j交互模式 ===")
    print("= 本系统将直接从Neo4j数据库查询医疗信息      =")
    print("= 输入 'quit' 或 'exit' 随时退出系统         =")
    print("= 输入 'help' 查看可用命令                   =")
    print()
    
    # 初始化Neo4j连接器
    try:
        from graph.neo4j_connector import Neo4jConnector
        
        # 尝试连接到Neo4j数据库
        print("正在连接Neo4j数据库...")
        with_loading_animation(lambda: time.sleep(1))
        
        # 可以从环境变量或配置文件读取连接信息
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # 获取连接信息（如果.env文件中有配置）
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        connector = Neo4jConnector(uri=uri, user=user, password=password)
        
        if not connector.driver:
            print(color_text("无法连接到Neo4j数据库，请检查连接信息。", "red"))
            return
        
        print(color_text("Neo4j数据库连接成功！", "green"))
        
    except ImportError:
        print(color_text("无法导入Neo4j连接器，请确保graph模块已正确安装。", "red"))
        return
    except Exception as e:
        print(color_text(f"初始化Neo4j连接时出错: {str(e)}", "red"))
        return
    
    try:
        while True:
            print("\n请输入命令:")
            command = input("neo4j> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("\n感谢使用医疗知识图谱系统，再见！")
                break
            
            if command in ['help', 'h', '?']:
                show_neo4j_help()
                continue
            
            # 解析命令
            parts = command.split()
            if not parts:
                continue
            
            cmd = parts[0]
            args = parts[1:]
            
            # 执行相应的命令
            if cmd == 'disease':
                handle_disease_query(connector, args)
            elif cmd == 'symptoms':
                handle_symptoms_query(connector, args)
            elif cmd == 'treatment':
                handle_treatment_query(connector, args)
            elif cmd == 'drugs':
                handle_drug_interaction_query(connector, args)
            elif cmd == 'schema':
                handle_schema_command(connector)
            else:
                print(color_text(f"未知命令: {cmd}", "red"))
                print("输入 'help' 查看可用命令")
    
    except KeyboardInterrupt:
        print("\n\n退出Neo4j交互模式")
    except Exception as e:
        print(color_text(f"\n错误: {str(e)}", "red"))
    finally:
        # 确保关闭数据库连接
        if 'connector' in locals():
            connector.close()
            print("\nNeo4j数据库连接已关闭")

def show_neo4j_help():
    """显示Neo4j交互模式的帮助信息"""
    print("\n可用命令:")
    print("  disease <疾病名称>      - 查询特定疾病的详细信息")
    print("  symptoms <症状1> <症状2>... - 根据症状查找相似疾病")
    print("  treatment <疾病名称>    - 查询特定疾病的治疗方案")
    print("  drugs <药物1> <药物2>... - 查询药物之间的相互作用")
    print("  schema                  - 初始化医疗知识图谱模式")
    print("  help/h/?                - 显示此帮助信息")
    print("  quit/exit/q             - 退出系统")

def handle_disease_query(connector, args):
    """处理疾病查询命令"""
    if len(args) < 1:
        print(color_text("请提供疾病名称", "red"))
        print("用法: disease <疾病名称>")
        return
    
    disease_name = ' '.join(args)
    print(f"\n正在查询疾病: {color_text(disease_name, 'green')}")
    
    try:
        # 查询疾病信息（这里假设数据库中有合适的查询方法）
        # 由于Neo4jConnector没有直接查询单个疾病的方法，我们可以先查询治疗路径来获取疾病信息
        treatment_paths = connector.find_treatment_paths(disease_name)
        
        if not treatment_paths:
            print(color_text(f"未找到疾病: {disease_name}", "yellow"))
            return
        
        # 显示疾病信息
        print(f"\n{color_text('疾病信息:', 'blue', bold=True)}")
        print(f"  名称: {treatment_paths[0]['disease']}")
        print(f"  治疗方案数量: {len(set(path['treatment'] for path in treatment_paths))}")
        
        # 显示治疗方案
        print(f"\n{color_text('推荐治疗方案:', 'blue', bold=True)}")
        
        # 按治疗方案分组
        treatments = {}
        for path in treatment_paths:
            if path['treatment'] not in treatments:
                treatments[path['treatment']] = path
        
        # 显示每个治疗方案
        for i, (treatment_name, path) in enumerate(treatments.items(), 1):
            print(f"\n{i}. {color_text(treatment_name, 'green')}")
            print(f"   类型: {path['treatment_type']}")
            print(f"   疗效: {path['efficacy']}")
            print(f"   证据级别: {path['evidence_level']}")
            print(f"   治疗线: {path['line_of_therapy']}")
            
            # 显示相关药物
            if path['drugs'] and path['drugs'][0]['drug']:
                print(f"   相关药物:")
                for drug in path['drugs']:
                    if drug['drug']:
                        print(f"     - {drug['drug']} ({drug['mechanism'] if 'mechanism' in drug and drug['mechanism'] else '未知机制'})")
    
    except Exception as e:
        print(color_text(f"查询疾病时出错: {str(e)}", "red"))

def handle_symptoms_query(connector, args):
    """处理症状查询命令"""
    if len(args) < 1:
        print(color_text("请至少提供一个症状", "red"))
        print("用法: symptoms <症状1> <症状2>...")
        return
    
    symptoms = args
    print(f"\n正在根据症状查找相似疾病: {color_text(', '.join(symptoms), 'green')}")
    
    try:
        # 调用Neo4jConnector的find_similar_cases方法
        similar_cases = connector.find_similar_cases(symptoms)
        
        if not similar_cases:
            print(color_text("未找到与这些症状相关的疾病", "yellow"))
            return
        
        # 显示结果
        print(f"\n{color_text('相似疾病（按匹配症状数量排序）:', 'blue', bold=True)}")
        
        for i, case in enumerate(similar_cases, 1):
            print(f"\n{i}. {color_text(case['disease'], 'green')}")
            print(f"   类别: {case['category']}")
            print(f"   匹配症状数: {color_text(f'{case['symptom_overlap']}/{len(symptoms)}', 'yellow')}")
            
            if case['treatments']:
                print(f"   治疗方案: {', '.join(case['treatments'][:3])}")
                if len(case['treatments']) > 3:
                    print(f"   ...等{len(case['treatments'])}个治疗方案")
    
    except Exception as e:
        print(color_text(f"查询症状时出错: {str(e)}", "red"))

def handle_treatment_query(connector, args):
    """处理治疗方案查询命令"""
    if len(args) < 1:
        print(color_text("请提供疾病名称", "red"))
        print("用法: treatment <疾病名称>")
        return
    
    disease_name = ' '.join(args)
    print(f"\n正在查询疾病的治疗方案: {color_text(disease_name, 'green')}")
    
    try:
        # 调用Neo4jConnector的find_treatment_paths方法
        treatment_paths = connector.find_treatment_paths(disease_name)
        
        if not treatment_paths:
            print(color_text(f"未找到疾病 '{disease_name}' 的治疗方案", "yellow"))
            return
        
        # 显示治疗路径
        print(f"\n{color_text('治疗方案（按治疗线和疗效排序）:', 'blue', bold=True)}")
        
        # 按治疗方案去重
        treatments_seen = set()
        for i, path in enumerate(treatment_paths, 1):
            if path['treatment'] not in treatments_seen:
                treatments_seen.add(path['treatment'])
                print(f"\n{i}. {color_text(path['treatment'], 'green')}")
                print(f"   类型: {path['treatment_type']}")
                print(f"   疗效评分: {color_text(f'{path['efficacy']}', 'yellow')}")
                print(f"   证据级别: {path['evidence_level']}")
                print(f"   治疗线: {path['line_of_therapy']}")
                
                # 显示药物信息
                if path['drugs'] and path['drugs'][0]['drug']:
                    print(f"   相关药物:")
                    for drug in path['drugs']:
                        if drug['drug']:
                            print(f"     - {drug['drug']}")
                            if 'mechanism' in drug and drug['mechanism']:
                                print(f"       作用机制: {drug['mechanism']}")
                            if 'dosage' in drug and drug['dosage']:
                                print(f"       剂量: {drug['dosage']}")
    
    except Exception as e:
        print(color_text(f"查询治疗方案时出错: {str(e)}", "red"))

def handle_drug_interaction_query(connector, args):
    """处理药物相互作用查询命令"""
    if len(args) < 2:
        print(color_text("请至少提供两种药物名称", "red"))
        print("用法: drugs <药物1> <药物2>...")
        return
    
    drug_names = args
    print(f"\n正在查询药物相互作用: {color_text(', '.join(drug_names), 'green')}")
    
    try:
        # 调用Neo4jConnector的get_drug_interactions方法
        interactions = connector.get_drug_interactions(drug_names)
        
        if not interactions:
            print(color_text("未发现这些药物之间的已知相互作用", "green"))
            return
        
        # 显示相互作用信息
        print(f"\n{color_text('药物相互作用信息:', 'blue', bold=True)}")
        
        for i, interaction in enumerate(interactions, 1):
            print(f"\n{i}. {color_text(interaction['drug1'], 'green')} 与 {color_text(interaction['drug2'], 'green')}")
            print(f"   相互作用类型: {interaction['interaction_type']}")
            print(f"   严重程度: {color_text(interaction['severity'], 'red' if 'high' in interaction['severity'].lower() else 'yellow')}")
            if 'description' in interaction and interaction['description']:
                print(f"   描述: {interaction['description']}")
    
    except Exception as e:
        print(color_text(f"查询药物相互作用时出错: {str(e)}", "red"))

def handle_schema_command(connector):
    """处理初始化数据库模式命令"""
    print("\n正在初始化医疗知识图谱模式...")
    
    try:
        # 调用Neo4jConnector的create_medical_schema方法
        connector.create_medical_schema()
        print(color_text("医疗知识图谱模式初始化成功！", "green"))
    except Exception as e:
        print(color_text(f"初始化数据库模式时出错: {str(e)}", "red"))

def get_enhanced_user_input() -> dict:
    """获取增强的用户输入"""
    
    print("\n请输入患者详细信息：")
    
    try:
        # 基本信息
        age = input("年龄: ").strip()
        if age.lower() in ['quit', 'exit']:
            return None
        
        gender = input("性别: ").strip()
        if gender.lower() in ['quit', 'exit']:
            return None
        
        # 可选信息
        print("\n基本信息已记录，现在请输入症状信息（必填）")
        symptoms_input = input("症状（用逗号分隔主要症状）: ").strip()
        if symptoms_input.lower() in ['quit', 'exit']:
            return None
        symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]
        
        print("\n以下信息为可选，输入 'skip' 可跳过")
        
        # 病史
        medical_history_input = input("既往病史（用逗号分隔）: ").strip()
        if medical_history_input.lower() in ['quit', 'exit']:
            return None
        medical_history = [] if medical_history_input.lower() == 'skip' else [
            h.strip() for h in medical_history_input.split(',') if h.strip()
        ]
        
        # 检查结果
        test_results = {}
        test_results_input = input("检查结果（格式：项目:结果，多个用分号分隔）: ").strip()
        if test_results_input.lower() in ['quit', 'exit']:
            return None
        elif test_results_input.lower() != 'skip':
            for item in test_results_input.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    test_results[key.strip()] = value.strip()
        
        # 影像学数据
        imaging_data = {}
        imaging_input = input("影像学数据（格式：类型:描述，多个用分号分隔）: ").strip()
        if imaging_input.lower() in ['quit', 'exit']:
            return None
        elif imaging_input.lower() != 'skip':
            for item in imaging_input.split(';'):
                if ':' in item:
                    key, value = item.split(':', 1)
                    imaging_data[key.strip()] = value.strip()
        
        return {
            'age': int(age) if age.isdigit() else 0,
            'gender': gender,
            'symptoms': symptoms,
            'medical_history': medical_history,
            'test_results': test_results,
            'imaging_data': imaging_data
        }
        
    except (ValueError, KeyboardInterrupt):
        return None

def create_case_from_enhanced_input(case_data: dict) -> MedicalCase:
    """从增强的用户输入创建案例"""
    
    return MedicalCase(
        case_id=f"interactive_{int(time.time())}",
        patient_info={
            'age': case_data['age'],
            'gender': case_data['gender']
        },
        symptoms=case_data['symptoms'],
        medical_history=case_data['medical_history'],
        test_results=case_data['test_results'],
        imaging_data=case_data['imaging_data']
    )

def color_text(text: str, color: str, bold: bool = False) -> str:
    """为文本添加终端颜色"""
    # Windows终端ANSI颜色代码
    colors = {
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'cyan': '36',
        'magenta': '35',
        'white': '37'
    }
    
    bold_code = '1' if bold else ''
    color_code = colors.get(color, '37')  # 默认白色
    
    return f"\033[{bold_code};{color_code}m{text}\033[0m"

def with_loading_animation(func, duration: float = 1.0):
    """在执行函数时显示加载动画"""
    import threading
    import sys
    
    stop_loading = False
    
    def loading_animation():
        chars = ['-', '/', '|', '\\']
        i = 0
        while not stop_loading:
            sys.stdout.write(f'\r处理中 {chars[i % len(chars)]} ')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write('\r')
        sys.stdout.flush()
    
    # 启动加载动画线程
    loading_thread = threading.Thread(target=loading_animation)
    loading_thread.daemon = True
    loading_thread.start()
    
    try:
        # 执行函数
        func()
    finally:
        # 停止加载动画
        stop_loading = True
        loading_thread.join(timeout=0.2)  # 等待动画线程结束
        sys.stdout.write('\r' + ' ' * 20 + '\r')
        sys.stdout.flush()

def create_default_config(config_file: str):
    """创建默认配置文件"""
    
    default_config = {
        "test_dataset_path": "data/test_cases.json",
        "clinical_guidelines_path": "data/clinical_guidelines.json",
        "output_dir": "experiment_results",
        "batch_size": 5,
        "include_agents": [
            "oncologist",
            "radiologist", 
            "nurse",
            "psychologist",
            "patient_advocate"
        ],
        "evaluation_metrics": [
            "diagnosis_accuracy",
            "treatment_appropriateness",
            "consensus_quality"
        ],
        "comparison_baselines": [
            "clinical_guidelines"
        ]
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    print(f"默认配置已保存到 {config_file}")

if __name__ == "__main__":
    main()