# backend/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio
import json
import logging
from datetime import datetime

from agents.base_agent import MedicalCase
from backend.api.chat import router as chat_router
from agents.oncologist import OncologistAgent
from agents.radiologist import RadiologistAgent
from agents.nurse import NurseAgent
from agents.patient_advocate import PatientAdvocateAgent
from agents.psychologist import PsychologistAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化智能体
oncologist = OncologistAgent()
radiologist = RadiologistAgent()
nurse = NurseAgent()
patient_advocate = PatientAdvocateAgent()
psychologist = PsychologistAgent()

# 创建一个简单的 MedicalCase 对象用于 WebSocket 通信
def create_simple_medical_case(message: str, history: list) -> MedicalCase:
    """从消息和历史创建简单的 MedicalCase 对象"""
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
    
    # 模拟从大模型提取的信息
    parsed_info = {
        "patient_info": {"message": message, "age": "未知", "gender": "未知"},
        "symptoms": [],
        "medical_history": [],
        "test_results": {},
        "imaging_data": {}
    }
    
    # 从对话历史中提取更多信息
    history_text = ""
    for msg in history:
        history_text += " " + str(msg)
    
    if history_text:
        parsed_info["patient_info"]["history"] = history_text.strip()
    
    return MedicalCase(
        case_id="websocket_case_" + str(hash(message))[-8:],
        patient_info=parsed_info["patient_info"],
        symptoms=parsed_info["symptoms"],
        medical_history=parsed_info["medical_history"],
        test_results=parsed_info["test_results"],
        imaging_data=parsed_info["imaging_data"]
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    logger.info("医疗AI多智能体对话系统已启动")
    logger.info(f"初始化智能体: {oncologist.agent_id}, {radiologist.agent_id}, {nurse.agent_id}, {patient_advocate.agent_id}, {psychologist.agent_id}")
    yield
    # 关闭事件
    logger.info("医疗AI多智能体对话系统正在关闭")

app = FastAPI(
    title="医疗AI多智能体对话系统",
    description="基于多智能体协作的医疗咨询AI系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置 - 优化配置以避免跨域请求问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"新的WebSocket连接建立，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接断开，当前连接数: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            await websocket.send_json(message)
            logger.debug(f"消息已发送: {message.get('type')}")
        except Exception as e:
            logger.error(f"发送消息失败: {str(e)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"广播消息失败: {str(e)}")

manager = ConnectionManager()

# 路由
app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    return {
        "message": "医疗AI多智能体对话系统",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "oncologist": "active",
            "radiologist": "active",
            "nurse": "active",
            "patient_advocate": "active",
            "psychologist": "active"
        }
    }


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        # 发送欢迎消息
        await manager.send_personal_message({
            "type": "system",
            "content": "欢迎使用医疗AI多智能体咨询系统！我是由多位专家智能体组成的医疗团队。",
            "timestamp": datetime.now().isoformat()
        }, websocket)

        while True:
            # 接收用户消息
            data = await websocket.receive_text()
            message_data = json.loads(data)

            user_message = message_data.get("message", "")
            conversation_history = message_data.get("history", [])

            logger.info(f"收到用户消息: {user_message[:50]}...")

            # 回显用户消息
            await manager.send_personal_message({
                "type": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            }, websocket)

            # 显示"思考中"状态
            await manager.send_personal_message({
                "type": "thinking",
                "content": "多学科专家正在分析您的问题...",
                "timestamp": datetime.now().isoformat()
            }, websocket)

            # 模拟异步处理
            await asyncio.sleep(1)

            # 肿瘤科医生分析
            await manager.send_personal_message({
                "type": "agent_thinking",
                "agent": "oncologist",
                "content": "🩺 肿瘤科医生正在分析...",
                "timestamp": datetime.now().isoformat()
            }, websocket)

            # 使用 agent 进行分析
            medical_case = create_simple_medical_case(user_message, conversation_history)
            onco_opinion = oncologist.analyze_case(medical_case)
            
            onco_response = {
                "content": onco_opinion.opinion,
                "confidence": onco_opinion.confidence,
                "recommendations": onco_opinion.recommendations
            }

            await manager.send_personal_message({
                "type": "agent_response",
                "agent": "oncologist",
                "agent_name": "肿瘤科医生",
                "content": onco_response["content"],
                "confidence": round(onco_response["confidence"], 2),
                "recommendations": onco_response.get("recommendations", []),
                "timestamp": datetime.now().isoformat()
            }, websocket)

            await asyncio.sleep(0.5)

            # 影像科医生分析
            await manager.send_personal_message({
                "type": "agent_thinking",
                "agent": "radiologist",
                "content": "🔍 影像科医生正在分析...",
                "timestamp": datetime.now().isoformat()
            }, websocket)

            # 使用 agent 进行分析
            radio_opinion = radiologist.analyze_case(medical_case)
            
            radio_response = {
                "content": radio_opinion.opinion,
                "confidence": radio_opinion.confidence,
                "recommendations": radio_opinion.recommendations
            }

            await manager.send_personal_message({
                "type": "agent_response",
                "agent": "radiologist",
                "agent_name": "影像科医生",
                "content": radio_response["content"],
                "confidence": round(radio_response["confidence"], 2),
                "recommendations": radio_response.get("recommendations", []),
                "timestamp": datetime.now().isoformat()
            }, websocket)

            await asyncio.sleep(0.5)

            # 护士分析
            await manager.send_personal_message({
                "type": "agent_thinking",
                "agent": "nurse",
                "content": "👩🏥 肿瘤专科护士正在分析...",
                "timestamp": datetime.now().isoformat()
            }, websocket)

            # 使用 agent 进行分析
            nurse_opinion = nurse.analyze_case(medical_case)
            
            nurse_response = {
                "content": nurse_opinion.opinion,
                "confidence": nurse_opinion.confidence,
                "recommendations": nurse_opinion.recommendations
            }

            await manager.send_personal_message({
                "type": "agent_response",
                "agent": "nurse",
                "agent_name": "肿瘤专科护士",
                "content": nurse_response["content"],
                "confidence": round(nurse_response["confidence"], 2),
                "recommendations": nurse_response.get("recommendations", []),
                "timestamp": datetime.now().isoformat()
            }, websocket)

            await asyncio.sleep(0.5)

            # 患者权益倡导者分析
            await manager.send_personal_message({
                "type": "agent_thinking",
                "agent": "patient_advocate",
                "content": "👨⚖️ 患者权益倡导者正在分析...",
                "timestamp": datetime.now().isoformat()
            }, websocket)

            # 使用 agent 进行分析
            advocate_opinion = patient_advocate.analyze_case(medical_case)
            
            advocate_response = {
                "content": advocate_opinion.opinion,
                "confidence": advocate_opinion.confidence,
                "recommendations": advocate_opinion.recommendations
            }

            await manager.send_personal_message({
                "type": "agent_response",
                "agent": "patient_advocate",
                "agent_name": "患者权益倡导者",
                "content": advocate_response["content"],
                "confidence": round(advocate_response["confidence"], 2),
                "recommendations": advocate_response.get("recommendations", []),
                "timestamp": datetime.now().isoformat()
            }, websocket)

            await asyncio.sleep(0.5)

            # 心理专家分析
            await manager.send_personal_message({
                "type": "agent_thinking",
                "agent": "psychologist",
                "content": "🧠 肿瘤心理专家正在分析...",
                "timestamp": datetime.now().isoformat()
            }, websocket)

            # 使用 agent 进行分析
            psych_opinion = psychologist.analyze_case(medical_case)
            
            psych_response = {
                "content": psych_opinion.opinion,
                "confidence": psych_opinion.confidence,
                "recommendations": psych_opinion.recommendations
            }

            await manager.send_personal_message({
                "type": "agent_response",
                "agent": "psychologist",
                "agent_name": "肿瘤心理专家",
                "content": psych_response["content"],
                "confidence": round(psych_response["confidence"], 2),
                "recommendations": psych_response.get("recommendations", []),
                "timestamp": datetime.now().isoformat()
            }, websocket)

            await asyncio.sleep(0.5)

            # 生成共识
            consensus = await generate_consensus(onco_response, radio_response, nurse_response, advocate_response, psych_response)

            await manager.send_personal_message({
                "type": "consensus",
                "content": consensus["content"],
                "confidence": round(consensus["confidence"], 2),
                "recommendations": consensus["recommendations"],
                "timestamp": datetime.now().isoformat()
            }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("用户断开了WebSocket连接")
    except json.JSONDecodeError:
        await manager.send_personal_message({
            "type": "error",
            "content": "消息格式错误，请重试。",
            "timestamp": datetime.now().isoformat()
        }, websocket)
    except Exception as e:
        logger.error(f"处理消息时发生错误: {str(e)}")
        await manager.send_personal_message({
            "type": "error",
            "content": "处理您的请求时发生错误，请稍后重试。",
            "timestamp": datetime.now().isoformat()
        }, websocket)
        manager.disconnect(websocket)


async def generate_consensus(oncologist_response: Dict[str, Any], radiologist_response: Dict[str, Any], 
                           nurse_response: Dict[str, Any], advocate_response: Dict[str, Any], 
                           psychologist_response: Dict[str, Any]) -> Dict[str, Any]:
    """生成多智能体共识"""
    # 计算平均置信度
    confidence = (oncologist_response["confidence"] + radiologist_response["confidence"] + 
                 nurse_response["confidence"] + advocate_response["confidence"] + 
                 psychologist_response["confidence"]) / 5
    
    # 合并推荐建议，去重并排序
    all_recommendations = (oncologist_response["recommendations"] + 
                          radiologist_response["recommendations"] + 
                          nurse_response["recommendations"] + 
                          advocate_response["recommendations"] + 
                          psychologist_response["recommendations"])
    
    # 去重 - 保持顺序
    seen = set()
    unique_recommendations = []
    for rec in all_recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    # 生成共识内容
    content = "# 🏥 多智能体专家共识意见\n\n"
    content += "尊敬的患者，基于肿瘤科医生、影像科医生、肿瘤专科护士、患者权益倡导者和肿瘤心理专家的专业分析，我们为您提供以下综合建议：\n\n"
    
    # 按诊断、治疗、随访分类整理建议
    diagnostic_recs = [r for r in unique_recommendations if any(keyword in r for keyword in ["检查", "诊断", "评估"])]
    treatment_recs = [r for r in unique_recommendations if any(keyword in r for keyword in ["治疗", "方案", "药物"])]
    followup_recs = [r for r in unique_recommendations if any(keyword in r for keyword in ["随访", "复查", "监测"])]
    care_recs = [r for r in unique_recommendations if any(keyword in r for keyword in ["护理", "症状", "副作用", "自我管理"])]
    support_recs = [r for r in unique_recommendations if any(keyword in r for keyword in ["支持", "资源", "权益", "沟通"])]
    mental_recs = [r for r in unique_recommendations if any(keyword in r for keyword in ["情绪", "心理", "应对", "放松"])]
    
    if diagnostic_recs:
        content += "## 📋 诊断建议\n"
        for i, rec in enumerate(diagnostic_recs, 1):
            content += f"{i}. {rec}\n"
        content += "\n"
    
    if treatment_recs:
        content += "## 💊 治疗建议\n"
        for i, rec in enumerate(treatment_recs, 1):
            content += f"{i}. {rec}\n"
        content += "\n"
    
    if care_recs:
        content += "## 👨⚕️ 护理指导\n"
        for i, rec in enumerate(care_recs, 1):
            content += f"{i}. {rec}\n"
        content += "\n"
    
    if support_recs:
        content += "## 🤝 支持与权益\n"
        for i, rec in enumerate(support_recs, 1):
            content += f"{i}. {rec}\n"
        content += "\n"
    
    if mental_recs:
        content += "## 🧠 心理健康\n"
        for i, rec in enumerate(mental_recs, 1):
            content += f"{i}. {rec}\n"
        content += "\n"
    
    if followup_recs:
        content += "## ⏱️ 随访计划\n"
        for i, rec in enumerate(followup_recs, 1):
            content += f"{i}. {rec}\n"
        content += "\n"
    
    # 添加综合健康建议
    content += "## 🌟 综合健康管理\n"
    content += "1. 保持积极乐观的心态，对疾病治疗充满信心\n"
    content += "2. 遵循医生建议，按时服药和进行复查\n"
    content += "3. 注意休息，保证充足睡眠，避免过度劳累\n"
    content += "4. 加强营养，保持均衡饮食，提高身体免疫力\n"
    content += "5. 适当进行体育锻炼，增强体质\n"
    content += "6. 积极寻求社会支持，与家人朋友保持沟通\n"
    content += "7. 关注心理健康，必要时寻求专业心理辅导\n\n"
    
    content += "*请注意：以上建议仅供参考，具体治疗方案请遵循您的主治医生的建议。如有任何疑问，请及时与医疗团队沟通。*"
    
    return {
        "content": content,
        "confidence": confidence,
        "recommendations": unique_recommendations[:7]  # 最多返回7条最重要的建议
    }

#  启动和关闭事件已移至 lifespan 上下文管理器中


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
