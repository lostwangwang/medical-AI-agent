# agents/radiologist.py
from datetime import datetime
from typing import Dict, Any
import json

from agents.base_agent import AgentRole, BaseAgent, MedicalCase, AgentOpinion


class RadiologistAgent(BaseAgent):
    def __init__(self, agent_id: str = "radiologist_001"):
        super().__init__(agent_id, AgentRole.RADIOLOGIST)
        self.expertise = [
            "肿瘤影像诊断", "CT/PET分析", "MRI解读", 
            "肿瘤分期评估", "治疗反应监测", "预后评估"
        ]
    
    def get_role_prompt(self) -> str:
        return """
        你是一位专业的放射科医生，专注于肿瘤的影像学诊断和评估。
        
        专业领域：
        - 肿瘤的影像学特征分析
        - CT、MRI、PET等多模态影像解读
        - 肿瘤大小、位置和边界评估
        - 肿瘤侵犯范围和转移情况分析
        - 肿瘤分期的影像学评估
        - 治疗前后的疗效对比
        - 肿瘤复发的早期识别
        
        诊断原则：
        1. 基于影像学表现的客观评估
        2. 结合临床信息的综合分析
        3. 遵循循证医学指南
        4. 注重肿瘤形态学和功能特征
        5. 提供准确的肿瘤分期
        """
    
    def _build_parse_prompt(self, raw_text):
        """
        辅助方法：构造二次解析Prompt
        :param raw_text: LLM初始生成的非结构化分析文本
        :return: 格式化的解析Prompt
        """
        return f"""
        任务：将以下影像学评估文本，提取为指定格式的JSON数据，用于填入结构化对象。
        请严格遵守以下要求：
        1. 仅返回JSON，不要包含任何额外文字（如解释、备注），否则会导致解析失败。
        2. JSON必须包含以下5个字段，字段类型和含义如下：
           - "confidence": 数值（0-1），代表评估信心度；若文本无明确值，填0.92。
           - "reasoning": 字符串，代表影像评估的核心依据；若无明确内容，填"基于多模态影像学表现的综合分析"。
           - "recommendations": 数组（字符串列表），代表具体影像检查建议（每项1个行动，如["增强CT扫描", "PET-CT检查"]）；若无明确建议，填["增强CT扫描", "PET-CT检查", "MRI增强扫描", "定期随访评估", "影像引导下活检"]。
           - "concerns": 数组（字符串列表），代表需关注的风险点（如["肿瘤边界不清", "多病灶可能"]）；若无明确风险，填["肿瘤边界不清", "多病灶可能", "远处转移迹象", "邻近器官侵犯", "治疗后变化不明确"]。
           - "priority_score": 数值（0-10），代表处理优先级；若文本无明确值，填9.0。
        3. 提取内容需严格对应原始文本，不添加文本外的信息。

        待解析的影像学评估文本：
        {raw_text}
        """

    def _parse_json_safely(self, json_str):
        """
        辅助方法：安全解析JSON
        :param json_str: LLM二次生成的JSON字符串
        :return: 解析后的字典
        """
        try:
            # 清理多余字符
            clean_str = json_str.strip().strip('"').strip("'").replace("\\n", "\n")
            return json.loads(clean_str)
        except json.JSONDecodeError:
            print(f"JSON解析失败，使用默认值。原始输入：{json_str}")
            return {
                "confidence": 0.92,
                "reasoning": "基于多模态影像学表现的综合分析",
                "recommendations": ["增强CT扫描", "PET-CT检查", "MRI增强扫描", "定期随访评估", "影像引导下活检"],
                "concerns": ["肿瘤边界不清", "多病灶可能", "远处转移迹象", "邻近器官侵犯", "治疗后变化不明确"],
                "priority_score": 9.0
            }
    
    def analyze_case(self, case: MedicalCase) -> AgentOpinion:
        """
        核心方法：输入病例数据，输出结构化的AgentOpinion
        """
        # 1. 构造初始影像评估Prompt
        initial_prompt = f"""
        {self.get_role_prompt()}

        请分析以下肿瘤病例并提供影像学评估：
        患者信息：{json.dumps(case.patient_info, ensure_ascii=False)}
        症状：{', '.join(case.symptoms)}
        病史：{', '.join(case.medical_history)}
        检查结果：{json.dumps(case.test_results, ensure_ascii=False)}
        影像资料：{json.dumps(case.imaging_data, ensure_ascii=False)}

        请从放射科医生的角度提供：
        1. 肿瘤影像学特征分析
        2. 肿瘤大小、位置和形态评估
        3. 肿瘤侵犯范围和转移情况
        4. 肿瘤分期的影像学判断
        5. 进一步影像检查建议

        请给出你的评估信心度(0-1)和优先级评分(0-10)。
        """
        # 调用LLM获取初始分析结果
        raw_analysis = self.llm_call(initial_prompt)
        print("影像评估初始分析结果：", raw_analysis)

        # 2. 构造二次解析Prompt
        parse_prompt = self._build_parse_prompt(raw_analysis)

        # 3. 调用LLM二次解析，获取结构化数据
        parsed_json_str = self.llm_call(parse_prompt)
        structured_data = self._parse_json_safely(parsed_json_str)

        # 4. 将结构化数据填入AgentOpinion
        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            opinion=raw_analysis,  # 保留原始分析文本
            confidence=structured_data["confidence"],  # 解析出的信心度
            reasoning=structured_data["reasoning"],  # 解析出的核心依据
            recommendations=structured_data["recommendations"],  # 解析出的建议列表
            concerns=structured_data["concerns"],  # 解析出的风险点列表
            priority_score=structured_data["priority_score"],  # 解析出的优先级
            timestamp=datetime.now().isoformat()  # 当前时间戳
        )
    
    def respond_to_question(self, question: str, context: Dict[str, Any]) -> str:
        prompt = f"""
        {self.get_role_prompt()}
        
        问题：{question}
        上下文：{json.dumps(context, ensure_ascii=False)}
        
        请以放射科医生的专业角度回答这个问题。
        """
        
        # 调用LLM
        response = self.llm_call(prompt)
        return response