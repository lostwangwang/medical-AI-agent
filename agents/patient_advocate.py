# agents/patient_advocate.py
from datetime import datetime
from typing import Dict, Any
import json

from agents.base_agent import AgentRole, BaseAgent, MedicalCase, AgentOpinion


class PatientAdvocateAgent(BaseAgent):
    def __init__(self, agent_id: str = "advocate_001"):
        super().__init__(agent_id, AgentRole.PATIENT_ADVOCATE)
        self.advocacy_focus = [
            "知情同意", "权益保障", "医患沟通", 
            "医疗资源协调", "治疗决策支持", "生活质量保障"
        ]
    
    def get_role_prompt(self) -> str:
        return """
        你是一位专业的患者权益倡导者，致力于保护肿瘤患者在医疗过程中的权益和尊严。
        
        专业领域：
        - 肿瘤患者的知情权和知情同意
        - 医患沟通的优化
        - 治疗决策的支持和引导
        - 医疗资源的合理利用
        - 治疗副作用的全程管理
        - 生活质量的持续改善
        - 患者心理和社会支持
        
        工作原则：
        1. 以患者为中心，尊重患者自主权
        2. 提供充分的信息和教育
        3. 促进医患有效沟通
        4. 确保患者得到公正的医疗服务
        5. 关注患者的整体生活质量
        """
    
    def _build_parse_prompt(self, raw_text):
        """
        辅助方法：构造二次解析Prompt
        :param raw_text: LLM初始生成的非结构化分析文本
        :return: 格式化的解析Prompt
        """
        return f"""
        任务：将以下患者权益评估文本，提取为指定格式的JSON数据，用于填入结构化对象。
        请严格遵守以下要求：
        1. 仅返回JSON，不要包含任何额外文字（如解释、备注），否则会导致解析失败。
        2. JSON必须包含以下5个字段，字段类型和含义如下：
           - "confidence": 数值（0-1），代表评估信心度；若文本无明确值，填0.90。
           - "reasoning": 字符串，代表权益评估的核心依据；若无明确内容，填"基于患者病情和权益需求的全面评估"。
           - "recommendations": 数组（字符串列表），代表具体权益保障建议（每项1个行动，如["确保知情同意完善", "优化医患沟通"]）；若无明确建议，填["确保知情同意完善", "优化医患沟通", "提供治疗决策支持", "协调医疗资源", "改善生活质量措施"]。
           - "concerns": 数组（字符串列表），代表需关注的风险点（如["治疗决策压力", "医患沟通障碍"]）；若无明确风险，填["治疗决策压力", "医患沟通障碍", "知情同意不充分", "医疗资源获取困难", "心理支持不足"]。
           - "priority_score": 数值（0-10），代表处理优先级；若文本无明确值，填8.0。
        3. 提取内容需严格对应原始文本，不添加文本外的信息。

        待解析的患者权益评估文本：
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
                "confidence": 0.90,
                "reasoning": "基于患者病情和权益需求的全面评估",
                "recommendations": ["确保知情同意完善", "优化医患沟通", "提供治疗决策支持", "协调医疗资源", "改善生活质量措施"],
                "concerns": ["治疗决策压力", "医患沟通障碍", "知情同意不充分", "医疗资源获取困难", "心理支持不足"],
                "priority_score": 8.0
            }
    
    def analyze_case(self, case: MedicalCase) -> AgentOpinion:
        """
        核心方法：输入病例数据，输出结构化的AgentOpinion
        """
        # 1. 构造初始患者权益评估Prompt
        initial_prompt = f"""
        {self.get_role_prompt()}

        请分析以下肿瘤病例并提供患者权益评估：
        患者信息：{json.dumps(case.patient_info, ensure_ascii=False)}
        症状：{', '.join(case.symptoms)}
        病史：{', '.join(case.medical_history)}
        检查结果：{json.dumps(case.test_results, ensure_ascii=False)}
        影像资料：{json.dumps(case.imaging_data, ensure_ascii=False)}

        请从患者权益倡导者的角度提供：
        1. 患者知情权保障评估
        2. 医患沟通状况分析
        3. 治疗决策支持建议
        4. 医疗资源协调方案
        5. 生活质量改善措施

        请给出你的评估信心度(0-1)和优先级评分(0-10)。
        """
        # 调用LLM获取初始分析结果
        raw_analysis = self.llm_call(initial_prompt)
        print("患者权益评估初始分析结果：", raw_analysis)

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
        
        请以患者权益倡导者的专业角度回答这个问题。
        """
        
        # 调用LLM
        response = self.llm_call(prompt)
        return response