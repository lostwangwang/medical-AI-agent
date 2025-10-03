# agents/oncologist.py
from typing import Dict, Any
from datetime import datetime
import json

from agents.base_agent import BaseAgent, AgentRole, MedicalCase, AgentOpinion


class OncologistAgent(BaseAgent):
    def __init__(self, agent_id: str = "onco_001"):
        super().__init__(agent_id, AgentRole.ONCOLOGIST)
        self.specialties = [
            "肿瘤诊断", "化疗方案", "放疗计划", "靶向治疗", 
            "免疫治疗", "预后评估", "分期诊断"
        ]
        
    def get_role_prompt(self) -> str:
        return """
        你是一位资深的肿瘤科医生，具有20年临床经验。
        
        专业领域：
        - 恶性肿瘤的诊断和分期
        - 制定个性化治疗方案
        - 化疗、放疗、靶向治疗的选择
        - 免疫治疗的适应症评估
        - 肿瘤预后评估
        - 临床试验的适用性评估
        
        分析原则：
        1. 基于循证医学证据
        2. 考虑患者整体状况
        3. 平衡治疗效果与副作用
        4. 考虑患者生活质量
        5. 遵循最新的肿瘤治疗指南
        
        请以专业、严谨的态度分析每个病例。
        """

    def _build_parse_prompt(self, raw_text):
        """
        辅助方法：构造二次解析Prompt（确保LLM理解输出格式）
        :param raw_text: LLM初始生成的非结构化分析文本
        :return: 格式化的解析Prompt
        """
        return f"""
        任务：将以下肿瘤病例分析文本，提取为指定格式的JSON数据，用于填入结构化对象。
        请严格遵守以下要求：
        1. 仅返回JSON，不要包含任何额外文字（如解释、备注），否则会导致解析失败。
        2. JSON必须包含以下5个字段，字段类型和含义如下：
           - "confidence": 数值（0-1），代表诊断信心度；若文本无明确值，填0.7。
           - "reasoning": 字符串，代表诊断/预后的核心依据（需包含影像学、实验室指标等关键逻辑）；若无明确内容，填"基于患者临床表现与检查结果综合分析"。
           - "recommendations": 数组（字符串列表），代表具体治疗/检查建议（每项1个行动，如["完善PET-CT检查", "进行基因检测"]）；若无明确建议，填["完善病理活检明确诊断", "组织多学科会诊制定方案"]。
           - "concerns": 数组（字符串列表），代表需关注的风险点（如["肿瘤远处转移风险", "患者体力状态评估"]）；若无明确风险，填["需进一步明确肿瘤分期", "密切监测患者基础疾病状态"]。
           - "priority_score": 数值（0-10），代表处理优先级；若文本无明确值，填8.0。
        3. 提取内容需严格对应原始文本，不添加文本外的信息。

        待解析的病例分析文本：
        {raw_text}
        """

    def _parse_json_safely(self, json_str):
        """
        辅助方法：安全解析JSON（处理LLM输出格式不标准的情况）
        :param json_str: LLM二次生成的JSON字符串
        :return: 解析后的字典（含默认值兜底）
        """
        try:
            # 清理多余字符（如LLM可能多带的引号、换行）
            clean_str = json_str.strip().strip('"').strip("'").replace("\\n", "\n")
            return json.loads(clean_str)
        except json.JSONDecodeError:
            # 解析失败时，返回合理默认值（避免程序崩溃）
            print(f"JSON解析失败，使用默认值。原始输入：{json_str}")
            return {
                "confidence": 0.7,
                "reasoning": "基于患者临床表现与检查结果综合分析",
                "recommendations": ["完善病理活检明确诊断", "组织多学科会诊制定方案"],
                "concerns": ["需进一步明确肿瘤分期", "密切监测患者基础疾病状态"],
                "priority_score": 8.0
            }

    def analyze_case(self, case):
        """
        核心方法：输入病例数据，输出结构化的AgentOpinion
        :param case: 病例对象（需包含patient_info、symptoms等属性）
        :return: AgentOpinion 结构化结果
        """
        # 1. 第一步：构造初始病例分析Prompt（复用你原有逻辑）
        initial_prompt = f"""
        {self.get_role_prompt()}

        请分析以下肿瘤病例：
        患者信息：{json.dumps(case.patient_info, ensure_ascii=False)}
        症状：{', '.join(case.symptoms)}
        病史：{', '.join(case.medical_history)}
        检查结果：{json.dumps(case.test_results, ensure_ascii=False)}
        影像资料：{json.dumps(case.imaging_data, ensure_ascii=False)}

        请从肿瘤科医生的角度提供：
        1. 诊断分析
        2. 分期评估
        3. 治疗建议
        4. 预后评估
        5. 需要关注的风险点

        请给出你的信心度(0-1)和优先级评分(0-10)。
        """
        # 调用LLM获取初始非结构化分析结果
        raw_analysis = self.llm_call(initial_prompt)

        # 2. 第二步：构造二次解析Prompt（关键：让LLM输出结构化JSON）
        parse_prompt = self._build_parse_prompt(raw_analysis)

        # 3. 第三步：调用LLM二次解析，获取结构化数据
        parsed_json_str = self.llm_call(parse_prompt)
        # 解析JSON并处理异常（核心容错逻辑）
        structured_data = self._parse_json_safely(parsed_json_str)

        # 4. 第四步：将结构化数据填入AgentOpinion
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
        
        请以肿瘤科医生的专业角度回答这个问题。
        """
        
        # 调用LLM
        response = self.llm_call(prompt)
        return response
