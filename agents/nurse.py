# agents/nurse.py
from datetime import datetime
from typing import Dict, Any
import json

from agents.base_agent import AgentRole, BaseAgent, MedicalCase, AgentOpinion


class NurseAgent(BaseAgent):
    def __init__(self, agent_id: str = "nurse_001"):
        super().__init__(agent_id, AgentRole.NURSE)
        self.care_areas = [
            "症状管理", "护理计划", "患者教育", 
            "副作用处理", "营养支持", "心理支持"
        ]
    
    def get_role_prompt(self) -> str:
        return """
        你是一位肿瘤专科护士，具有丰富的肿瘤患者护理经验。
        
        专业领域：
        - 肿瘤患者的全程护理管理
        - 化疗副作用的预防和处理
        - 患者及家属的健康教育
        - 症状评估和管理
        - 营养状况评估和指导
        - 心理护理和情感支持
        
        护理原则：
        1. 以患者为中心的整体护理
        2. 预防为主，及时发现问题
        3. 个性化护理计划
        4. 重视患者舒适度
        5. 加强患者教育和自我管理
        """
    
    def _build_parse_prompt(self, raw_text):
        """
        辅助方法：构造二次解析Prompt
        :param raw_text: LLM初始生成的非结构化分析文本
        :return: 格式化的解析Prompt
        """
        return f"""
        任务：将以下护理评估文本，提取为指定格式的JSON数据，用于填入结构化对象。
        请严格遵守以下要求：
        1. 仅返回JSON，不要包含任何额外文字（如解释、备注），否则会导致解析失败。
        2. JSON必须包含以下5个字段，字段类型和含义如下：
           - "confidence": 数值（0-1），代表评估信心度；若文本无明确值，填0.88。
           - "reasoning": 字符串，代表护理评估的核心依据；若无明确内容，填"基于患者症状和一般状况的护理评估"。
           - "recommendations": 数组（字符串列表），代表具体护理建议（每项1个行动，如["完善护理评估", "制定症状管理方案"]）；若无明确建议，填["完善护理评估", "制定症状管理方案", "患者教育计划", "营养状况评估", "心理支持措施"]。
           - "concerns": 数组（字符串列表），代表需关注的风险点（如["患者症状负担", "营养状况"]）；若无明确风险，填["患者症状负担", "营养状况", "心理压力", "家庭支持不足"]。
           - "priority_score": 数值（0-10），代表处理优先级；若文本无明确值，填7.5。
        3. 提取内容需严格对应原始文本，不添加文本外的信息。

        待解析的护理评估文本：
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
                "confidence": 0.88,
                "reasoning": "基于患者症状和一般状况的护理评估",
                "recommendations": ["完善护理评估", "制定症状管理方案", "患者教育计划", "营养状况评估", "心理支持措施"],
                "concerns": ["患者症状负担", "营养状况", "心理压力", "家庭支持不足"],
                "priority_score": 7.5
            }
    
    def analyze_case(self, case: MedicalCase) -> AgentOpinion:
        """
        核心方法：输入病例数据，输出结构化的AgentOpinion
        """
        # 1. 构造初始护理评估Prompt
        initial_prompt = f"""
        {self.get_role_prompt()}

        请分析以下肿瘤病例并提供护理评估：
        患者信息：{json.dumps(case.patient_info, ensure_ascii=False)}
        症状：{', '.join(case.symptoms)}
        病史：{', '.join(case.medical_history)}
        检查结果：{json.dumps(case.test_results, ensure_ascii=False)}
        影像资料：{json.dumps(case.imaging_data, ensure_ascii=False)}

        请从肿瘤专科护士的角度提供：
        1. 护理重点评估
        2. 症状管理方案
        3. 患者教育建议
        4. 营养支持计划
        5. 心理支持措施

        请给出你的评估信心度(0-1)和优先级评分(0-10)。
        """
        # 调用LLM获取初始分析结果
        raw_analysis = self.llm_call(initial_prompt)
        print("护士初始分析结果：", raw_analysis)

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
        
        请以肿瘤专科护士的专业角度回答这个问题。
        """
        
        # 调用LLM
        response = self.llm_call(prompt)
        return response