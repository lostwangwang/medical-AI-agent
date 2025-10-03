# backend/api/chat.py
import json
import os
import asyncio

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from agents.base_agent import MedicalCase, AgentOpinion
from agents.nurse import NurseAgent
from agents.oncologist import OncologistAgent
from agents.patient_advocate import PatientAdvocateAgent
from agents.psychologist import PsychologistAgent
from agents.radiologist import RadiologistAgent

router = APIRouter()
logger = logging.getLogger(__name__)

# 创建智能体实例
oncologist = OncologistAgent()
radiologist = RadiologistAgent()
nurse = NurseAgent()
patient_advocate = PatientAdvocateAgent()
psychologist = PsychologistAgent()


def _llm_call(extraction_prompt):
    # 确保API密钥已设置
    if not os.getenv("QWEN_API_KEY"):
        logger.warning("API密钥未设置，无法调用大模型")
        return ""
    
    try:
        client = OpenAI(
            # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
            api_key=os.getenv("QWEN_API_KEY"),
            # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[{"role": "user", "content": extraction_prompt}],
        )
        target_message = response.choices[0].message
        content_data = target_message.content
        return content_data
    except Exception as e:
        logger.error(f"大模型调用失败: {str(e)}")
        return ""

# 辅助函数：将用户消息转换为MedicalCase对象
def create_medical_case_from_message(message: str, conversation_history: List[Dict[str, Any]]) -> MedicalCase:
    """从用户消息和对话历史创建MedicalCase对象"""
    # 使用大模型从用户消息中提取结构化信息
    extraction_prompt = f"""
    请从以下患者描述中提取结构化的医疗信息，并以JSON格式返回：
    
    患者描述：{message}
    
    请提取以下信息：
    1. 患者基本信息（年龄、性别等）
    2. 主要症状（症状类型、持续时间、严重程度等）
    3. 病史（既往病史、家族史、过敏史等）
    4. 检查结果（实验室检查、病理检查等）
    5. 影像学资料（CT、MRI、X光等检查结果）
    
    请严格按照以下JSON格式返回，未提及的信息请留空：
    {{
        "patient_info": {{
            "age": "年龄（数字或'未知'）",
            "gender": "性别（男/女/未知）",
            "other": "其他基本信息"
        }},
        "symptoms": ["症状1", "症状2", ...],
        "medical_history": ["病史1", "病史2", ...],
        "test_results": {{
            "test_name": "结果"
        }},
        "imaging_data": {{
            "imaging_type": "结果描述"
        }}
    }}
    """
    
    # 从对话历史中提取更多信息
    history_text = ""
    for msg in conversation_history:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            history_text += " " + content
    
    # 在实际应用中，应该使用类似下面的代码：
    try:
        extracted_info = _llm_call(extraction_prompt)
        # 检查返回的内容是否为空
        if extracted_info and extracted_info.strip():
            parsed_info = json.loads(extracted_info)
            
            if history_text:
                parsed_info["patient_info"]["history"] = history_text.strip()
        else:
            raise ValueError("大模型返回空内容")
    except Exception as e:
        logger.warning(f"大模型调用失败，使用默认数据: {str(e)}")
        # 使用默认数据
        parsed_info = {
            "patient_info": {"message": message, "age": "未知", "gender": "未知"},
            "symptoms": [],
            "medical_history": [],
            "test_results": {},
            "imaging_data": {}
        }
        
        if history_text:
            parsed_info["patient_info"]["history"] = history_text.strip()
    
    return MedicalCase(
        case_id="api_case_" + str(hash(message))[-8:],
        patient_info=parsed_info["patient_info"],
        symptoms=parsed_info["symptoms"],
        medical_history=parsed_info["medical_history"],
        test_results=parsed_info["test_results"],
        imaging_data=parsed_info["imaging_data"]
    )

# 辅助函数：将AgentOpinion转换为API响应格式
def convert_opinion_to_api_response(opinion: AgentOpinion, agent_type: str) -> Dict[str, Any]:
    """将AgentOpinion转换为API响应格式"""
    # 确保 opinion 是 AgentOpinion 实例
    if not isinstance(opinion, AgentOpinion):
        return {
            "content": "分析需要更多信息。",
            "confidence": 0.7,
            "recommendations": [],
            "analysis_type": agent_type
        }
    
    base_response = {
        "content": getattr(opinion, "opinion", "无内容"),
        "confidence": getattr(opinion, "confidence", 0.5),
        "recommendations": getattr(opinion, "recommendations", []),
        "analysis_type": agent_type
    }
    
    # 根据智能体类型添加特定字段
    if agent_type == "oncological":
        base_response["key_findings"] = getattr(opinion, "key_findings", [])
    elif agent_type == "radiological":
        base_response["imaging_suggestions"] = getattr(opinion, "imaging_suggestions", [])
    elif agent_type == "nursing":
        base_response["symptom_management"] = getattr(opinion, "symptom_management", [])
    elif agent_type == "advocacy":
        base_response["rights_information"] = getattr(opinion, "rights_information", [])
    elif agent_type == "psychological":
        base_response["coping_strategies"] = getattr(opinion, "coping_strategies", [])
    
    return base_response


class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


# 添加到文件顶部的导入部分
import uuid
from fastapi import BackgroundTasks
import time

# 添加到文件中，用于存储处理中的请求
processing_requests = {}

# 添加一个新的数据模型用于异步响应
class AsyncResponse(BaseModel):
    request_id: str
    initial_response: str
    status: str = "processing"
    message: str = "您的请求已接收，医疗团队正在后台分析中..."

@router.post("/chat/async", response_model=AsyncResponse)
async def chat_async(chat_message: ChatMessage, background_tasks: BackgroundTasks) -> AsyncResponse:
    """异步聊天接口，立即返回初步响应，后台处理完整分析"""
    try:
        logger.info(f"收到异步聊天请求: {chat_message.message[:50]}...")
        
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())
        
        # 提取历史记录（如果有）
        history = []
        if chat_message.conversation_history:
            history = chat_message.conversation_history
        
        # 创建医疗案例
        medical_case = create_medical_case_from_message(chat_message.message, history)
        
        # 生成初步响应
        initial_response = f"感谢您的咨询。我们的医疗团队正在分析您的问题: '{chat_message.message[:50]}...'\n\n"
        initial_response += "我们将为您提供以下专家的分析意见：\n"
        initial_response += "• 肿瘤科医生\n• 影像科医生\n• 肿瘤专科护士\n• 患者权益倡导者\n• 肿瘤心理专家\n\n"
        initial_response += "请稍候，我们会实时更新分析结果..."
        
        # 初始化请求状态
        processing_requests[request_id] = {
            "status": "processing",
            "message": chat_message.message,
            "history": history,
            "agent_responses": {},
            "consensus": None,
            "created_at": time.time()
        }
        
        # 添加后台任务
        background_tasks.add_task(process_chat_in_background, request_id, medical_case)
        
        return AsyncResponse(
            request_id=request_id,
            initial_response=initial_response
        )
        
    except Exception as e:
        logger.error(f"处理异步聊天请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理您的请求时发生错误: {str(e)}")

async def process_chat_in_background(request_id: str, medical_case: MedicalCase):
    """后台处理完整的医疗分析"""
    try:
        # 更新状态为处理中
        processing_requests[request_id]["status"] = "analyzing"
        
        # 调用各个智能体进行分析
        logger.info(f"为请求 {request_id} 开始调用各个智能体进行分析...")
        
        # 逐个调用智能体，这样可以更新进度
        onco_response = await analyze_with_oncologist_api(medical_case)
        processing_requests[request_id]["agent_responses"]["oncologist"] = onco_response
        processing_requests[request_id]["status"] = "20%"
        
        radio_response = await analyze_with_radiologist_api(medical_case)
        processing_requests[request_id]["agent_responses"]["radiologist"] = radio_response
        processing_requests[request_id]["status"] = "40%"
        
        nurse_response = await analyze_with_nurse_api(medical_case)
        processing_requests[request_id]["agent_responses"]["nurse"] = nurse_response
        processing_requests[request_id]["status"] = "60%"
        
        advocate_response = await analyze_with_advocate_api(medical_case)
        processing_requests[request_id]["agent_responses"]["patient_advocate"] = advocate_response
        processing_requests[request_id]["status"] = "80%"
        
        psych_response = await analyze_with_psychologist_api(medical_case)
        processing_requests[request_id]["agent_responses"]["psychologist"] = psych_response
        processing_requests[request_id]["status"] = "90%"
        
        # 生成共识
        consensus = await generate_consensus_api(
            onco_response, radio_response, nurse_response, advocate_response, psych_response
        )
        processing_requests[request_id]["consensus"] = consensus
        
        # 更新状态为完成
        processing_requests[request_id]["status"] = "completed"
        logger.info(f"请求 {request_id} 处理完成")
        
    except Exception as e:
        logger.error(f"后台处理请求 {request_id} 时发生错误: {str(e)}")
        processing_requests[request_id]["status"] = "error"
        processing_requests[request_id]["error"] = str(e)

@router.get("/chat/status/{request_id}")
async def get_chat_status(request_id: str) -> Dict[str, Any]:
    """获取聊天请求的处理状态和结果"""
    if request_id not in processing_requests:
        raise HTTPException(status_code=404, detail="请求ID不存在")
    
    request_data = processing_requests[request_id]
    
    # 构建响应
    response = {
        "request_id": request_id,
        "status": request_data["status"],
        "created_at": request_data["created_at"],
        "agent_responses": request_data["agent_responses"],
        "consensus": request_data.get("consensus")
    }
    
    # 如果有错误，添加错误信息
    if "error" in request_data:
        response["error"] = request_data["error"]
    
    return response

class ChatResponse(BaseModel):
    response: str
    agent_responses: List[dict]
    consensus: dict
    timestamp: str


@router.post("/chat")
async def chat(chat_message: ChatMessage) -> ChatResponse:
    try:
        logger.info(f"收到API聊天请求: {chat_message.message[:50]}...")
        # 提取历史记录（如果有）
        history = []
        if chat_message.conversation_history:
            history = chat_message.conversation_history
        medical_case = create_medical_case_from_message(chat_message.message, history)
        
        # 调用各个智能体进行分析
        logger.info("开始调用各个智能体进行分析...")
        # 并行调用所有智能体
        tasks = [
            analyze_with_oncologist_api(medical_case),
            analyze_with_radiologist_api(medical_case),
            analyze_with_nurse_api(medical_case),
            analyze_with_advocate_api(medical_case),
            analyze_with_psychologist_api(medical_case)
        ]
        
        # 等待所有任务完成
        onco_response, radio_response, nurse_response, advocate_response, psych_response = await asyncio.gather(*tasks)
        
        # 生成共识
        consensus = await generate_consensus_api(
            onco_response, radio_response, nurse_response, advocate_response, psych_response
        )
        
        # 构建响应
        agent_responses = [
            {"agent": "oncologist", "agent_name": "肿瘤科医生", "content": onco_response.get("content", "无内容"), "confidence": onco_response.get("confidence", 0.5), "recommendations": onco_response.get("recommendations", [])},
            {"agent": "radiologist", "agent_name": "影像科医生", "content": radio_response.get("content", "无内容"), "confidence": radio_response.get("confidence", 0.5), "recommendations": radio_response.get("recommendations", [])},
            {"agent": "nurse", "agent_name": "肿瘤专科护士", "content": nurse_response.get("content", "无内容"), "confidence": nurse_response.get("confidence", 0.5), "recommendations": nurse_response.get("recommendations", [])},
            {"agent": "patient_advocate", "agent_name": "患者权益倡导者", "content": advocate_response.get("content", "无内容"), "confidence": advocate_response.get("confidence", 0.5), "recommendations": advocate_response.get("recommendations", [])},
            {"agent": "psychologist", "agent_name": "肿瘤心理专家", "content": psych_response.get("content", "无内容"), "confidence": psych_response.get("confidence", 0.5), "recommendations": psych_response.get("recommendations", [])}
        ]
        
        return ChatResponse(
            response=consensus["content"],
            agent_responses=agent_responses,
            consensus={"content": consensus["content"], "confidence": consensus["confidence"], "recommendations": consensus["recommendations"]},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"处理聊天请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理您的请求时发生错误: {str(e)}")


@router.get("/chat/history/{conversation_id}")
async def get_chat_history(conversation_id: str) -> Dict[str, Any]:
    """获取对话历史"""
    try:
        # 实际项目中应该从数据库获取历史记录
        # 暂时返回模拟数据
        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": "user",
                    "content": "我最近感觉胸部不适，想咨询一下可能的原因。",
                    "timestamp": datetime.now().isoformat(),
                    "message_id": "msg_001"
                },
                {
                    "role": "agent",
                    "content": "作为多智能体医疗团队，我们建议您尽快进行相关检查...",
                    "timestamp": datetime.now().isoformat(),
                    "message_id": "msg_002",
                    "agent_id": "consensus"
                }
            ],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取对话历史时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取对话历史失败: {str(e)}")


# API版本的智能体分析函数
@router.post("/chat/agent/{agent_type}")
async def analyze_with_agent(chat_message: ChatMessage, agent_type: str) -> dict:
    """单独请求特定智能体的分析"""
    try:
        logger.info(f"收到API智能体分析请求: agent_type={agent_type}, message={chat_message.message[:30]}...")
        
        # 提取历史记录（如果有）
        history = []
        if chat_message.conversation_history:
            history = chat_message.conversation_history
        
        medical_case = create_medical_case_from_message(chat_message.message, history)
        
        # 根据智能体类型调用对应的分析函数
        if agent_type == "oncologist":
            result = await analyze_with_oncologist_api(medical_case)
            result["agent_type"] = "oncologist"
            result["agent_name"] = "肿瘤科医生"
        elif agent_type == "radiologist":
            result = await analyze_with_radiologist_api(medical_case)
            result["agent_type"] = "radiologist"
            result["agent_name"] = "影像科医生"
        elif agent_type == "nurse":
            result = await analyze_with_nurse_api(medical_case)
            result["agent_type"] = "nurse"
            result["agent_name"] = "肿瘤专科护士"
        elif agent_type == "patient_advocate":
            result = await analyze_with_advocate_api(medical_case)
            result["agent_type"] = "patient_advocate"
            result["agent_name"] = "患者权益倡导者"
        elif agent_type == "psychologist":
            result = await analyze_with_psychologist_api(medical_case)
            result["agent_type"] = "psychologist"
            result["agent_name"] = "肿瘤心理专家"
        else:
            raise HTTPException(status_code=400, detail=f"不支持的智能体类型: {agent_type}")
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"智能体分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"智能体分析失败: {str(e)}")
async def analyze_with_oncologist_api(medical_case: MedicalCase) -> dict:
    """API版肿瘤科医生分析"""
    try:
        # 直接使用智能体进行分析
        opinion = oncologist.analyze_case(medical_case)
        return convert_opinion_to_api_response(opinion, "oncological")
    except Exception as e:
        logger.error(f"肿瘤科医生API分析错误: {str(e)}")
        return {
            "content": "肿瘤科医生分析过程中出现错误，请稍后重试。",
            "confidence": 0.3,
            "recommendations": ["请稍后重新提交您的问题"],
            "analysis_type": "oncological"
        }


async def analyze_with_radiologist_api(medical_case: MedicalCase) -> dict:
    """API版影像科医生分析"""
    try:
        # 直接使用智能体进行分析
        opinion = radiologist.analyze_case(medical_case)
        return convert_opinion_to_api_response(opinion, "radiological")
    except Exception as e:
        logger.error(f"影像科医生API分析错误: {str(e)}")
        return {
            "content": "影像科医生分析过程中出现错误，请稍后重试。",
            "confidence": 0.3,
            "recommendations": ["请稍后重新提交您的问题"],
            "analysis_type": "radiological"
        }


async def analyze_with_nurse_api(medical_case: MedicalCase) -> dict:
    """API版护士分析"""
    try:
        # 直接使用智能体进行分析
        opinion = nurse.analyze_case(medical_case)
        return convert_opinion_to_api_response(opinion, "nursing")
    except Exception as e:
        logger.error(f"护士API分析错误: {str(e)}")
        return {
            "content": "护士分析过程中出现错误，请稍后重试。",
            "confidence": 0.3,
            "recommendations": ["请稍后重新提交您的问题"],
            "analysis_type": "nursing"
        }


async def analyze_with_advocate_api(medical_case: MedicalCase) -> dict:
    """API版患者权益倡导者分析"""
    try:
        # 直接使用智能体进行分析
        opinion = patient_advocate.analyze_case(medical_case)
        return convert_opinion_to_api_response(opinion, "advocacy")
    except Exception as e:
        logger.error(f"患者权益倡导者API分析错误: {str(e)}")
        return {
            "content": "患者权益倡导者分析过程中出现错误，请稍后重试。",
            "confidence": 0.3,
            "recommendations": ["请稍后重新提交您的问题"],
            "analysis_type": "advocacy"
        }


async def analyze_with_psychologist_api(medical_case: MedicalCase) -> dict:
    """API版心理专家分析"""
    try:
        # 直接使用智能体进行分析
        opinion = psychologist.analyze_case(medical_case)
        return convert_opinion_to_api_response(opinion, "psychological")
    except Exception as e:
        logger.error(f"心理专家API分析错误: {str(e)}")
        return {
            "content": "心理专家分析过程中出现错误，请稍后重试。",
            "confidence": 0.3,
            "recommendations": ["请稍后重新提交您的问题"],
            "analysis_type": "psychological"
        }


async def generate_consensus_api(oncologist_response: Dict[str, Any], radiologist_response: Dict[str, Any], 
                                nurse_response: Dict[str, Any], advocate_response: Dict[str, Any], 
                                psychologist_response: Dict[str, Any]) -> Dict[str, Any]:
    """API版多智能体共识生成"""
    try:
        # 计算平均置信度
        confidence = (oncologist_response.get("confidence", 0) + radiologist_response.get("confidence", 0) + 
                    nurse_response.get("confidence", 0) + advocate_response.get("confidence", 0) + 
                    psychologist_response.get("confidence", 0)) / 5
        
        # 合并推荐建议，去重
        all_recommendations = (oncologist_response.get("recommendations", []) + 
                            radiologist_response.get("recommendations", []) + 
                            nurse_response.get("recommendations", []) + 
                            advocate_response.get("recommendations", []) + 
                            psychologist_response.get("recommendations", []))
        
        # 去重
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        # 生成共识内容
        content = "# 多智能体医疗团队共识意见\n\n"
        content += "基于我们多学科团队（肿瘤科医生、影像科医生、肿瘤专科护士、患者权益倡导者和心理专家）的综合分析：\n\n"
        
        # 添加各专家的核心观点
        content += f"**肿瘤科医生观点**：{oncologist_response.get('content', '无内容').split('\n')[0] if oncologist_response.get('content') else '无内容'}\n\n"
        content += f"**影像科医生观点**：{radiologist_response.get('content', '无内容').split('\n')[0] if radiologist_response.get('content') else '无内容'}\n\n"
        content += f"**肿瘤专科护士建议**：{nurse_response.get('content', '无内容').split('\n')[0] if nurse_response.get('content') else '无内容'}\n\n"
        content += f"**患者权益倡导者提示**：{advocate_response.get('content', '无内容').split('\n')[0] if advocate_response.get('content') else '无内容'}\n\n"
        content += f"**心理专家建议**：{psychologist_response.get('content', '无内容').split('\n')[0] if psychologist_response.get('content') else '无内容'}\n\n"
        
        # 添加综合建议
        content += "## 综合建议\n"
        for i, rec in enumerate(unique_recommendations[:7], 1):
            content += f"{i}. {rec}\n"
        
        content += "\n*请注意：以上建议仅供参考，具体治疗方案请遵循您的主治医生的建议。*"
        
        return {
            "content": content,
            "confidence": confidence,
            "recommendations": unique_recommendations[:7],
            "analysis_type": "consensus"
        }
    except Exception as e:
        logger.error(f"生成共识时发生错误: {str(e)}")
        return {
            "content": "生成多智能体共识时发生错误，请稍后重试。",
            "confidence": 0.4,
            "recommendations": ["请联系您的医疗团队获取专业建议"],
            "analysis_type": "consensus"
        }