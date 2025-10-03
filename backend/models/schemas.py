# backend/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class Message(BaseModel):
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    agent: Optional[str] = None
    message_id: Optional[str] = None
    conversation_history: Optional[List['Message']] = None

# 自引用更新
Message.model_rebuild()

class AgentResponse(BaseModel):
    agent_name: str
    agent_type: str
    content: str
    confidence: float
    recommendations: List[str]
    timestamp: datetime

class ConsensusResult(BaseModel):
    summary: str
    confidence: float
    recommendations: List[str]
    participating_agents: List[str]
    timestamp: datetime

class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime

class ChatMessage(BaseModel):
    """用于API聊天请求的消息模型"""
    message: str
    conversation_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    message_id: Optional[str] = None

class ChatResponse(BaseModel):
    """用于API聊天响应的模型"""
    response: str
    agent_responses: List[Dict[str, Any]]
    consensus: Dict[str, Any]
    timestamp: str