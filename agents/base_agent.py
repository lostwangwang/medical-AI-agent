# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

class AgentRole(Enum):
    ONCOLOGIST = "oncologist"
    RADIOLOGIST = "radiologist" 
    NURSE = "nurse"
    PSYCHOLOGIST = "psychologist"
    PATIENT_ADVOCATE = "patient_advocate"

@dataclass
class MedicalCase:
    case_id: str
    patient_info: Dict[str, Any]
    symptoms: List[str]
    medical_history: List[str]
    test_results: Dict[str, Any]
    imaging_data: Dict[str, Any]
    current_treatment: Optional[str] = None

@dataclass
class AgentOpinion:
    agent_id: str
    role: AgentRole
    opinion: str
    confidence: float  # 0-1
    reasoning: str
    recommendations: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    priority_score: float = 5.0  # 0-10
    timestamp: str = field(default_factory=lambda: str(os.path.getmtime(__file__)))

class BaseAgent(ABC):
    def __init__(self, agent_id: str, role: AgentRole, model_name: str = "qwen2.5-med"):
        self.agent_id = agent_id
        self.role = role
        self.model_name = model_name
        self.api_key = os.getenv("QWEN_API_KEY")
        self.knowledge_base = None
        self.conversation_history = []
        self.logger = logging.getLogger(f"{role.value}_{agent_id}")
        
    @abstractmethod
    def analyze_case(self, case: MedicalCase) -> AgentOpinion:
        """分析医疗案例并给出意见"""
        pass
    
    @abstractmethod
    def respond_to_question(self, question: str, context: Dict[str, Any]) -> str:
        """回答问题"""
        pass
    
    @abstractmethod
    def get_role_prompt(self) -> str:
        """获取角色提示词"""
        pass
    
    def update_knowledge_base(self, knowledge_base):
        """更新知识库"""
        self.knowledge_base = knowledge_base
    
    def log_interaction(self, interaction: Dict[str, Any]):
        """记录交互"""
        self.conversation_history.append(interaction)
        self.logger.info(f"Interaction logged: {interaction}")
        
    def llm_call(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        调用大语言模型进行文本生成
        
        Args:
            prompt: 提示文本
            temperature: 生成温度，控制输出多样性
            max_tokens: 最大生成长度
            
        Returns:
            模型生成的文本
        """
        try:
            # 确保API密钥已设置
            if not self.api_key:
                raise ValueError("API密钥未设置，请在.env文件中配置QWEN_API_KEY")
            
            # 清理提示文本中的无效字符，避免UTF-8编码问题
            def clean_text(text):
                # 去除或替换无法用UTF-8编码的字符
                return ''.join(c for c in text if c.isprintable() or c in '\n\t\r' or ord(c) >= 32)
            
            cleaned_prompt = clean_text(prompt)
            
            client = OpenAI(
                # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
                api_key=os.getenv("QWEN_API_KEY"),
                # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            response = client.chat.completions.create(
                model="qwen3-max",
                messages=[{"role": "user", "content": cleaned_prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            target_message = response.choices[0].message
            content_data = target_message.content
            return content_data
            
        except Exception as e:
            self.logger.error(f"LLM调用失败: {str(e)}")
            # 返回错误信息或默认响应
            return f"大语言模型调用失败: {str(e)}"