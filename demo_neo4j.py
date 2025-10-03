#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医疗知识图谱 - Neo4j数据库交互演示程序

此程序提供命令行界面，用于从Neo4j数据库查询医疗信息，包括疾病详情、症状匹配、治疗方案和药物相互作用。
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from graph.neo4j_connector import Neo4jConnector
except ImportError:
    print("无法导入Neo4j连接器，请确保graph模块已正确安装。")
    sys.exit(1)

def color_text(text: str, color: str = 'default', bold: bool = False) -> str:
    """为文本添加颜色和粗体"""
    colors = {
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'cyan': '36',
        'default': '39'
    }
    
    prefix = '\033['
    if bold:
        prefix += '1;'
    prefix += f'{colors.get(color, colors["default"])}m'
    
    return f"{prefix}{text}\033[0m"

def print_header():
    """打印程序头部信息"""
    header = """
    ################################################################
    #                  医疗知识图谱 - Neo4j演示程序               #
    #                                                              #
    #  本程序演示如何从Neo4j数据库查询医疗知识图谱数据。          #
    #  提供疾病信息、症状匹配、治疗方案和药物相互作用查询功能。    #
    #                                                              #
    #  输入 'help' 查看可用命令，'quit' 退出程序。                #
    ################################################################
    """
    print(color_text(header, 'cyan'))

def show_help():
    """显示帮助信息"""
    help_text = """
可用命令:
  disease <疾病名称>      查询特定疾病的详细信息
  symptoms <症状1> <症状2>... 根据症状查找相关疾病
  treatment <疾病名称>    查询特定疾病的治疗方案
  drugs <药物1> <药物2>... 查询药物之间的相互作用
  schema                  初始化医疗知识图谱模式
  help/h/?                显示此帮助信息
  quit/exit/q             退出程序
  """
    print(help_text)

def wait_for_keypress():
    """等待用户按键继续"""
    input("\n按Enter键继续...")

def demo_disease_query(connector):
    """演示疾病查询功能"""
    print(color_text("\n=== 疾病信息查询演示 ===", "blue", bold=True))
    print("本功能允许您查询特定疾病的详细信息和相关治疗方案。")
    
    # 示例疾病
    example_diseases = ["肺癌", "乳腺癌", "糖尿病"]
    
    print("\n示例疾病:")
    for i, disease in enumerate(example_diseases, 1):
        print(f"  {i}. {disease}")
    
    choice = input("\n请选择示例疾病(1-3)或输入其他疾病名称: ")
    
    if choice.isdigit() and 1 <= int(choice) <= len(example_diseases):
        disease_name = example_diseases[int(choice) - 1]
    else:
        disease_name = choice.strip() if choice.strip() else "肺癌"
    
    print(f"\n正在查询疾病: {color_text(disease_name, 'green')}")
    
    try:
        # 查询疾病的治疗路径来获取疾病信息
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
        
        # 显示前3个治疗方案
        for i, (treatment_name, path) in enumerate(list(treatments.items())[:3], 1):
            print(f"\n{i}. {color_text(treatment_name, 'green')}")
            print(f"   类型: {path['treatment_type']}")
            print(f"   疗效: {path['efficacy']}")
            print(f"   证据级别: {path['evidence_level']}")
            print(f"   治疗线: {path['line_of_therapy']}")
            
            # 显示相关药物
            if path['drugs'] and path['drugs'][0]['drug']:
                print(f"   相关药物:")
                for drug in path['drugs'][:2]:
                    if drug['drug']:
                        print(f"     - {drug['drug']}")
    
    except Exception as e:
        print(color_text(f"查询疾病时出错: {str(e)}", "red"))

def demo_symptoms_query(connector):
    """演示症状查询功能"""
    print(color_text("\n=== 症状匹配查询演示 ===", "blue", bold=True))
    print("本功能允许您根据症状查找可能的相关疾病。")
    
    # 示例症状组合
    example_symptoms = [
        ["咳嗽", "胸痛", "呼吸困难"],  # 可能与肺癌相关
        ["乳房肿块", "乳头溢液"],    # 可能与乳腺癌相关
        ["多饮", "多尿", "体重减轻"]  # 可能与糖尿病相关
    ]
    
    print("\n示例症状组合:")
    for i, symptoms in enumerate(example_symptoms, 1):
        print(f"  {i}. {', '.join(symptoms)}")
    
    choice = input("\n请选择示例症状组合(1-3)或输入您的症状(空格分隔): ")
    
    if choice.isdigit() and 1 <= int(choice) <= len(example_symptoms):
        symptoms = example_symptoms[int(choice) - 1]
    else:
        symptoms = [sym.strip() for sym in choice.split() if sym.strip()]
        if not symptoms:
            symptoms = ["咳嗽", "胸痛"]
    
    print(f"\n正在根据症状查找相似疾病: {color_text(', '.join(symptoms), 'green')}")
    
    try:
        # 调用find_similar_cases方法
        similar_cases = connector.find_similar_cases(symptoms)
        
        if not similar_cases:
            print(color_text("未找到与这些症状相关的疾病", "yellow"))
            return
        
        # 显示结果
        print(f"\n{color_text('相似疾病（按匹配症状数量排序）:', 'blue', bold=True)}")
        
        for i, case in enumerate(similar_cases[:5], 1):
            print(f"\n{i}. {color_text(case['disease'], 'green')}")
            print(f"   类别: {case['category']}")
            print(f"   匹配症状数: {color_text(f'{case['symptom_overlap']}/{len(symptoms)}', 'yellow')}")
            
            if case['treatments']:
                print(f"   治疗方案: {', '.join(case['treatments'][:2])}")
    
    except Exception as e:
        print(color_text(f"查询症状时出错: {str(e)}", "red"))

def demo_treatment_query(connector):
    """演示治疗方案查询功能"""
    print(color_text("\n=== 治疗方案查询演示 ===", "blue", bold=True))
    print("本功能允许您查询特定疾病的治疗方案和相关信息。")
    
    # 示例疾病
    example_diseases = ["肺癌", "乳腺癌", "糖尿病"]
    
    print("\n示例疾病:")
    for i, disease in enumerate(example_diseases, 1):
        print(f"  {i}. {disease}")
    
    choice = input("\n请选择示例疾病(1-3)或输入其他疾病名称: ")
    
    if choice.isdigit() and 1 <= int(choice) <= len(example_diseases):
        disease_name = example_diseases[int(choice) - 1]
    else:
        disease_name = choice.strip() if choice.strip() else "肺癌"
    
    print(f"\n正在查询疾病的治疗方案: {color_text(disease_name, 'green')}")
    
    try:
        # 调用find_treatment_paths方法
        treatment_paths = connector.find_treatment_paths(disease_name)
        
        if not treatment_paths:
            print(color_text(f"未找到疾病 '{disease_name}' 的治疗方案", "yellow"))
            return
        
        # 显示治疗路径
        print(f"\n{color_text('治疗方案（按治疗线和疗效排序）:', 'blue', bold=True)}")
        
        # 按治疗方案去重
        treatments_seen = set()
        for i, path in enumerate(treatment_paths[:5], 1):
            if path['treatment'] not in treatments_seen:
                treatments_seen.add(path['treatment'])
                print(f"\n{i}. {color_text(path['treatment'], 'green')}")
                print(f"   类型: {path['treatment_type']}")
                print(f"   疗效评分: {color_text(f'{path['efficacy']}', 'yellow')}")
                print(f"   证据级别: {path['evidence_level']}")
                print(f"   治疗线: {path['line_of_therapy']}")
    
    except Exception as e:
        print(color_text(f"查询治疗方案时出错: {str(e)}", "red"))

def demo_drug_interaction_query(connector):
    """演示药物相互作用查询功能"""
    print(color_text("\n=== 药物相互作用查询演示 ===", "blue", bold=True))
    print("本功能允许您查询药物之间可能的相互作用。")
    
    # 示例药物组合
    example_drug_pairs = [
        ["阿司匹林", "华法林"],      # 可能有重要的相互作用
        ["布洛芬", "对乙酰氨基酚"],  # 常见的疼痛药物组合
        ["他汀类药物", "红霉素"]    # 可能的代谢相互作用
    ]
    
    print("\n示例药物组合:")
    for i, drugs in enumerate(example_drug_pairs, 1):
        print(f"  {i}. {', '.join(drugs)}")
    
    choice = input("\n请选择示例药物组合(1-3)或输入您的药物(空格分隔): ")
    
    if choice.isdigit() and 1 <= int(choice) <= len(example_drug_pairs):
        drug_names = example_drug_pairs[int(choice) - 1]
    else:
        drug_names = [drug.strip() for drug in choice.split() if drug.strip()]
        if len(drug_names) < 2:
            drug_names = ["阿司匹林", "华法林"]
    
    print(f"\n正在查询药物相互作用: {color_text(', '.join(drug_names), 'green')}")
    
    try:
        # 调用get_drug_interactions方法
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

def run_demo_mode(connector):
    """运行演示模式"""
    print_header()
    
    while True:
        print("\n=== 请选择演示功能 ===")
        print("  1. 疾病信息查询")
        print("  2. 症状匹配查询")
        print("  3. 治疗方案查询")
        print("  4. 药物相互作用查询")
        print("  5. 交互式查询模式")
        print("  0. 退出程序")
        
        choice = input("\n请选择功能 (0-5): ").strip()
        
        if choice == "0":
            print("\n感谢使用医疗知识图谱演示程序，再见！")
            break
        elif choice == "1":
            demo_disease_query(connector)
            wait_for_keypress()
        elif choice == "2":
            demo_symptoms_query(connector)
            wait_for_keypress()
        elif choice == "3":
            demo_treatment_query(connector)
            wait_for_keypress()
        elif choice == "4":
            demo_drug_interaction_query(connector)
            wait_for_keypress()
        elif choice == "5":
            run_interactive_mode(connector)
        else:
            print("无效的选择，请重试。")

def run_interactive_mode(connector):
    """运行交互式查询模式"""
    print("\n=== 进入交互式查询模式 ===")
    print("输入 'help' 查看可用命令，'back' 返回主菜单，'quit' 退出程序。")
    
    while True:
        try:
            command = input("\nneo4j> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("\n感谢使用医疗知识图谱系统，再见！")
                exit(0)
            
            if command in ['back', 'b']:
                return
            
            if command in ['help', 'h', '?']:
                show_help()
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
            print("\n")
            continue
        except Exception as e:
            print(color_text(f"\n错误: {str(e)}", "red"))

def handle_disease_query(connector, args):
    """处理疾病查询命令"""
    if len(args) < 1:
        print(color_text("请提供疾病名称", "red"))
        print("用法: disease <疾病名称>")
        return
    
    disease_name = ' '.join(args)
    print(f"\n正在查询疾病: {color_text(disease_name, 'green')}")
    
    try:
        # 查询疾病信息
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
                for drug in path['drugs'][:3]:
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
        
        for i, case in enumerate(similar_cases[:5], 1):
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
        for i, path in enumerate(treatment_paths[:10], 1):
            if path['treatment'] not in treatments_seen:
                treatments_seen.add(path['treatment'])
                print(f"\n{i}. {color_text(path['treatment'], 'green')}")
                print(f"   类型: {path['treatment_type']}")
                print(f"   疗效评分: {color_text(f'{path['efficacy']}', 'yellow')}")
                print(f"   证据级别: {path['evidence_level']}")
                print(f"   治疗线: {path['line_of_therapy']}")
    
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

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="医疗知识图谱 - Neo4j数据库交互演示程序")
    parser.add_argument("--mode", choices=["demo", "interactive"], 
                        default="demo", help="运行模式")
    parser.add_argument("--uri", help="Neo4j数据库URI")
    parser.add_argument("--user", help="Neo4j用户名")
    parser.add_argument("--password", help="Neo4j密码")
    parser.add_argument("--env-file", default=".env", help="环境变量文件路径")
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv(args.env_file)
    
    # 获取连接信息
    uri = args.uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = args.user or os.getenv("NEO4J_USER", "neo4j")
    password = args.password or os.getenv("NEO4J_PASSWORD", "password")
    
    # 连接Neo4j数据库
    print(f"正在连接Neo4j数据库: {uri}")
    try:
        connector = Neo4jConnector(uri=uri, user=user, password=password)
        
        if not connector.driver:
            print(color_text("无法连接到Neo4j数据库，请检查连接信息。", "red"))
            sys.exit(1)
        
        print(color_text("Neo4j数据库连接成功！", "green"))
        
        # 运行指定模式
        if args.mode == "interactive":
            print_header()
            run_interactive_mode(connector)
        else:
            run_demo_mode(connector)
            
    except Exception as e:
        print(color_text(f"初始化Neo4j连接时出错: {str(e)}", "red"))
        sys.exit(1)
    finally:
        # 确保关闭数据库连接
        if 'connector' in locals():
            connector.close()
            print("\nNeo4j数据库连接已关闭")

if __name__ == "__main__":
    main()