import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import random
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MedicalDialogue:
    role: str
    content: str
    case_context: Dict[str, Any]

@dataclass
class TrainingExample:
    input_text: str
    output_text: str
    role: str
    case_id: str
    
class MedicalDataProcessor:
    """医疗数据处理器"""
    
    def __init__(self):
        self.role_templates = {
            'oncologist': {
                'system_prompt': '''你是一位资深的肿瘤科医生，具有20年临床经验。请基于提供的病例信息，从专业的肿瘤学角度进行分析。''',
                'few_shot_examples': [
                    {
                        'input': '患者，女性，45岁，发现右乳肿块2个月，钼靶显示4cm不规则肿块，核磁共振提示腋窝淋巴结肿大。',
                        'output': '''基于提供的临床信息，这是一个高度怀疑恶性的乳腺病变。
                        
分析要点：
1. 影像学特征：4cm不规则肿块提示恶性可能性大
2. 淋巴结受累：腋窝淋巴结肿大提示可能的区域转移
3. 临床分期：初步评估为T2N1M0，IIA-IIB期

推荐诊疗方案：
1. 立即行空心针穿刺活检确诊
2. 免疫组化检测ER、PR、HER2状态
3. 完善分期检查：胸腹CT、骨扫描
4. 多学科会诊制定综合治疗方案

预后评估：早期乳腺癌，及时治疗预后良好。'''
                    }
                ]
            },
            'radiologist': {
                'system_prompt': '''你是一位经验丰富的影像科医生，专精于肿瘤影像诊断。请详细分析影像学表现并给出专业意见。''',
                'few_shot_examples': [
                    {
                        'input': '胸部CT：右上肺见3.2cm结节状阴影，边缘不光滑，有毛刺征，增强扫描明显强化。',
                        'output': '''影像学分析：

CT表现：
- 病灶位置：右上肺
- 大小形态：3.2cm结节，形态不规则
- 边缘特征：毛刺征阳性，高度提示恶性
- 强化模式：明显强化，符合恶性肿瘤血供丰富特点

影像学诊断：右上肺恶性肿瘤，考虑肺癌可能性大

建议：
1. PET-CT评估代谢活性及全身转移情况
2. 气管镜检查或CT引导下穿刺活检
3. 胸部增强MRI评估纵隔侵犯
4. 腹部CT排除腹腔转移

鉴别诊断：需与炎性假瘤、转移瘤鉴别。'''
                    }
                ]
            }
        }
    
    def load_medical_cases(self, cases_file: str) -> List[Dict[str, Any]]:
        """加载医疗病例数据"""
        with open(cases_file, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        return cases
    
    def generate_roleplay_data(self, cases: List[Dict[str, Any]], 
                             num_examples_per_case: int = 5) -> List[TrainingExample]:
        """生成角色扮演训练数据"""
        training_examples = []
        
        for case in cases:
            case_id = case.get('case_id', '')
            
            # 为每个角色生成训练样本
            for role in self.role_templates.keys():
                for _ in range(num_examples_per_case):
                    example = self._generate_single_roleplay_example(case, role)
                    if example:
                        training_examples.append(example)
        
        return training_examples
    
    def _generate_single_roleplay_example(self, case: Dict[str, Any], role: str) -> TrainingExample:
        """生成单个角色扮演样本"""
        
        # 构建输入文本
        input_parts = []
        
        # 添加角色系统提示
        system_prompt = self.role_templates[role]['system_prompt']
        input_parts.append(f"<|system|>\n{system_prompt}")
        
        # 添加病例信息
        case_info = self._format_case_info(case)
        input_parts.append(f"<|user|>\n病例信息：\n{case_info}\n\n请从{role}的角度分析这个病例。")
        
        input_text = "\n\n".join(input_parts)
        
        # 生成输出文本（这里需要实际的专家标注数据，这里是示例）
        output_text = self._generate_role_response(case, role)
        
        return TrainingExample(
            input_text=input_text,
            output_text=output_text,
            role=role,
            case_id=case.get('case_id', '')
        )
    
    def _format_case_info(self, case: Dict[str, Any]) -> str:
        """格式化病例信息"""
        info_parts = []
        
        # 患者基本信息
        patient_info = case.get('patient_info', {})
        if patient_info:
            age = patient_info.get('age', '未知')
            gender = patient_info.get('gender', '未知')
            info_parts.append(f"患者信息：{gender}，{age}岁")
        
        # 主诉和症状
        symptoms = case.get('symptoms', [])
        if symptoms:
            info_parts.append(f"主要症状：{', '.join(symptoms)}")
        
        # 病史
        history = case.get('medical_history', [])
        if history:
            info_parts.append(f"既往史：{', '.join(history)}")
        
        # 检查结果
        test_results = case.get('test_results', {})
        if test_results:
            results_text = []
            for test, result in test_results.items():
                results_text.append(f"{test}: {result}")
            info_parts.append(f"检查结果：{'; '.join(results_text)}")
        
        # 影像资料
        imaging = case.get('imaging_data', {})
        if imaging:
            imaging_text = []
            for modality, findings in imaging.items():
                imaging_text.append(f"{modality}: {findings}")
            info_parts.append(f"影像学：{'; '.join(imaging_text)}")
        
        return "\n".join(info_parts)
    
    def _generate_role_response(self, case: Dict[str, Any], role: str) -> str:
        """生成角色回应（示例，实际需要专家标注）"""
        
        # 这里应该有真实的专家标注数据
        # 为了示例，我们生成模板化的回应
        
        if role == 'oncologist':
            return '''从肿瘤科医生的角度分析：

诊断考虑：基于临床表现和检查结果，高度怀疑恶性肿瘤。

分期评估：需要进一步完善检查明确分期。

治疗建议：
1. 完善病理组织学检查
2. 基因检测评估靶向治疗可能性
3. 多学科会诊制定综合治疗方案
4. 评估患者一般状况和手术耐受性

预后评估：需要根据最终病理结果和分期情况评估。'''
        
        elif role == 'radiologist':
            return '''影像科医生分析：

影像学表现：符合恶性病变的影像学特征。

进一步检查建议：
1. 增强扫描评估血供情况
2. 多序列MRI评估局部侵犯
3. PET-CT评估代谢活性和全身情况
4. 必要时行影像引导下活检

影像学随访：建议制定规范化的影像学随访方案。'''
        
        # 其他角色的模板回应...
        
        return "需要进一步分析..."
    
    def create_preference_dataset(self, roleplay_data: List[TrainingExample]) -> List[Dict[str, Any]]:
        """创建偏好数据集用于DPO训练"""
        preference_data = []
        
        # 按case_id和role分组
        grouped_data = {}
        for example in roleplay_data:
            key = f"{example.case_id}_{example.role}"
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(example)
        
        # 为每组生成偏好对
        for key, examples in grouped_data.items():
            if len(examples) >= 2:
                # 随机选择两个示例作为偏好对
                for i in range(len(examples)):
                    for j in range(i + 1, len(examples)):
                        # 随机决定哪个是preferred
                        if random.random() > 0.5:
                            preferred = examples[i]
                            rejected = examples[j]
                        else:
                            preferred = examples[j]
                            rejected = examples[i]
                        
                        preference_data.append({
                            'prompt': preferred.input_text,
                            'chosen': preferred.output_text,
                            'rejected': rejected.output_text,
                            'role': preferred.role,
                            'case_id': preferred.case_id
                        })
        
        return preference_data
    
    def save_training_data(self, data: List[TrainingExample], output_file: str):
        """保存训练数据"""
        formatted_data = []
        for example in data:
            formatted_data.append({
                'instruction': example.input_text,
                'output': example.output_text,
                'role': example.role,
                'case_id': example.case_id
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(formatted_data)} training examples to {output_file}")