<!-- frontend/src/App.vue -->
<template>
  <div id="app" class="app-container">
    <header class="app-header">
      <div class="header-content">
        <h1>🏥 医疗AI多智能体咨询系统</h1>
        <div class="header-status">
          <span
            class="status-indicator"
            :class="{ connected: isConnected }"
          ></span>
          <span>{{ isConnected ? "已连接" : "未连接" }}</span>
        </div>
      </div>
    </header>

    <div class="main-content">
      <!-- 智能体状态面板 -->
      <aside class="agents-panel">
        <h3>🤖 智能体团队</h3>
        <div class="agent-card" v-for="agent in agents" :key="agent.id">
          <div class="agent-avatar">{{ agent.emoji }}</div>
          <div class="agent-info">
            <h4>{{ agent.name }}</h4>
            <p class="agent-specialty">{{ agent.specialty }}</p>
            <div class="agent-status" :class="agent.status">
              {{ agent.statusText }}
            </div>
          </div>
        </div>
      </aside>

      <!-- 聊天窗口 -->
      <main class="chat-container">
        <ChatWindow
          :messages="messages"
          :is-loading="isLoading"
          @scroll-to-bottom="scrollToBottom"
        />

        <InputBox
          @send-message="sendMessage"
          :disabled="!isConnected || isLoading"
          :placeholder="inputPlaceholder"
        />
      </main>

      <!-- 建议面板 -->
      <aside class="recommendations-panel">
        <h3>💡 推荐建议</h3>
        <div
          v-if="currentRecommendations.length > 0"
          class="recommendations-list"
        >
          <div
            v-for="(rec, index) in currentRecommendations"
            :key="index"
            class="recommendation-item"
          >
            <span class="rec-number">{{ index + 1 }}</span>
            <span class="rec-text">{{ rec }}</span>
          </div>
        </div>
        <div v-else class="no-recommendations">
          <p>暂无推荐建议</p>
          <p class="hint">开始咨询后，专家团队会给出相应建议</p>
        </div>
      </aside>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted, onUnmounted, computed, nextTick } from "vue";
import ChatWindow from "./components/ChatWindow.vue";
import InputBox from "./components/InputBox.vue";
import axios from "axios";

// 创建axios实例
const apiClient = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 600000, // 增加超时时间到600秒
  headers: {
    "Content-Type": "application/json",
  },
});

export default {
  name: "App",
  components: {
    ChatWindow,
    InputBox,
  },
  setup() {
    const isConnected = ref(false);
    const isLoading = ref(false);
    const messages = ref([]);
    const currentRecommendations = ref([]);
    const healthCheckInterval = ref(null);

    const agents = reactive([
      {
        id: "oncologist",
        name: "肿瘤科医生",
        emoji: "🩺",
        specialty: "肿瘤诊断与治疗",
        status: "idle",
        statusText: "待命中",
      },
      {
        id: "radiologist",
        name: "影像科医生",
        emoji: "🔍",
        specialty: "医学影像分析",
        status: "idle",
        statusText: "待命中",
      },
      {
        id: "nurse",
        name: "肿瘤专科护士",
        emoji: "👩",
        specialty: "护理管理与症状支持",
        status: "idle",
        statusText: "待命中",
      },
      {
        id: "patient_advocate",
        name: "患者权益倡导者",
        emoji: "👨",
        specialty: "权益保障与决策支持",
        status: "idle",
        statusText: "待命中",
      },
      {
        id: "psychologist",
        name: "肿瘤心理专家",
        emoji: "🧠",
        specialty: "心理评估与干预",
        status: "idle",
        statusText: "待命中",
      },
    ]);

    const inputPlaceholder = computed(() => {
      if (!isConnected.value) return "正在连接服务器...";
      if (isLoading.value) return "专家正在分析中...";
      return "请描述您的症状或问题...";
    });

    // 健康检查函数 - 优化版本，增加错误处理和重试机制
    const checkHealth = async () => {
      try {
        // 重置连接状态为false，只有在成功响应后才设置为true
        isConnected.value = false;
        
        console.log('正在进行健康检查...');
        const response = await apiClient.get("/health");
        console.log('健康检查响应:', response.data);
        
        if (response.data.status === "healthy") {
          isConnected.value = true;

          // 更新智能体状态
          if (response.data.agents) {
            Object.entries(response.data.agents).forEach(
              ([agentId, status]) => {
                const agent = agents.find((a) => a.id === agentId);
                console.log(`更新智能体${agentId}状态为${status}`)
                if (agent) {
                  agent.status = status === "active" ? "idle" : status;
                  agent.statusText = status === "active" ? "待命中" : status;
                }
              }
            );
          }

          // 首次连接成功后发送一条系统消息
          if (messages.value.length === 0) {
            messages.value.push({
              id: Date.now(),
              type: "system",
              content: "已成功连接到医疗AI多智能体咨询系统！",
              timestamp: new Date().toISOString(),
            });
          }
        }
      } catch (error) {
        console.error("健康检查失败:", error);
        isConnected.value = false;
        
        // 显示连接错误消息
        if (!messages.value.some(msg => msg.type === "error" && msg.content.includes("连接失败"))) {
          messages.value.push({
            id: Date.now(),
            type: "error",
            content: "连接到服务器失败，请检查服务是否正常运行。",
            timestamp: new Date().toISOString(),
          });
        }
      }
    };

    // 优化历史消息，只发送相关的用户消息和智能体回复
    const getRelevantHistory = () => {
      return messages.value
        .filter(
          (msg) =>
            msg.type === "user" ||
            msg.type === "agent" ||
            msg.type === "consensus"
        )
        .slice(-10) // 只保留最近10条相关消息
        .map((msg) => ({
          role: msg.type, // 使用role而不是type与后端模型兼容
          content: msg.content,
          agent: msg.agent,
          timestamp: msg.timestamp,
        }));
    };

// 发送消息 - 逐个智能体分析方式
    const sendMessageSequentialAgents = async (message) => {
      if (!message.trim() || !isConnected.value) return;

      isLoading.value = true;
      
      try {
        // 添加用户消息到界面
        messages.value.push({
          id: Date.now(),
          type: "user",
          content: message.trim(),
          timestamp: new Date().toISOString(),
        });
        
        // 滚动到底部
        await nextTick();
        scrollToBottom();

        // 获取相关历史记录
        const relevantHistory = getRelevantHistory();
        
        const requestData = {
          message: message,
          conversation_history: relevantHistory,
        };
        
        // 所有智能体类型
        const agentTypes = [
          { id: "oncologist", name: "肿瘤科医生" },
          { id: "radiologist", name: "影像科医生" },
          { id: "nurse", name: "肿瘤专科护士" },
          { id: "patient_advocate", name: "患者权益倡导者" },
          { id: "psychologist", name: "肿瘤心理专家" }
        ];
        
        // 存储所有智能体的响应结果
        const agentResponses = [];
        
        // 逐个请求智能体分析
        for (const agent of agentTypes) {
          try {
            // 更新智能体状态为思考中
            updateAgentStatus(agent.id, "thinking", "分析中...");
            
            console.log(`开始请求${agent.name}分析...`);
            
            // 调用单个智能体分析API
            const response = await apiClient.post(`/chat/agent/${agent.id}`, requestData);
            const agentData = response.data;
            
            console.log(`${agent.name}分析完成:`, agentData);
            
            // 更新智能体状态为已完成
            updateAgentStatus(agent.id, "completed", "已完成");
            
            // 添加智能体回复到消息列表
            messages.value.push({
              id: Date.now(),
              type: "agent",
              agent: agent.id,
              agentName: agent.name,
              content: agentData.content || "无内容",
              confidence: agentData.confidence || 0.5,
              recommendations: agentData.recommendations || [],
              timestamp: new Date().toISOString(),
            });
            
            // 保存响应用于后续生成共识
            agentResponses.push({
              agent: agent.id,
              agent_name: agent.name,
              content: agentData.content,
              confidence: agentData.confidence,
              recommendations: agentData.recommendations
            });
            
            // 每次添加消息后滚动到底部
            await nextTick();
            scrollToBottom();
            
          } catch (error) {
            console.error(`${agent.name}分析失败:`, error);
            
            // 更新智能体状态为错误
            updateAgentStatus(agent.id, "error", "分析失败");
            
            // 添加错误消息
            messages.value.push({
              id: Date.now(),
              type: "error",
              content: `${agent.name}分析失败，请稍后重试`,
              timestamp: new Date().toISOString(),
            });
          }
        }
        
        // 所有智能体分析完成后，生成共识
        if (agentResponses.length > 0) {
          try {
            // 添加共识分析消息
            messages.value.push({
              id: Date.now(),
              type: "thinking",
              content: "正在整合专家意见，生成最终共识...",
              timestamp: new Date().toISOString(),
            });
            
            await nextTick();
            scrollToBottom();
            
            // 可以调用生成共识的API，这里暂时使用前端处理
            // 如果需要后端处理，可以添加一个专门的共识生成API
            const consensus = generateConsensusFromResponses(agentResponses);
            
            // 更新推荐建议
            currentRecommendations.value = consensus.recommendations || [];
            
            // 添加共识消息
            messages.value.push({
              id: Date.now(),
              type: "consensus",
              content: consensus.content || "无法生成共识意见",
              confidence: consensus.confidence || 0.5,
              timestamp: new Date().toISOString(),
            });
            
          } catch (error) {
            console.error("生成共识失败:", error);
            messages.value.push({
              id: Date.now(),
              type: "error",
              content: "生成专家共识失败，请稍后重试",
              timestamp: new Date().toISOString(),
            });
          }
        }
      } finally {
        isLoading.value = false;
        
        // 重置智能体状态（延迟执行，让用户能看到完成状态）
        setTimeout(() => {
          agents.forEach((agent) => {
            updateAgentStatus(agent.id, "idle", "就绪");
          });
        }, 3000);
        
        // 确保滚动到底部
        await nextTick();
        scrollToBottom();
      }
    };
    
    // 从智能体响应生成共识（前端简化版本）
    const generateConsensusFromResponses = (agentResponses) => {
      // 计算平均置信度
      const confidence = agentResponses.reduce((sum, agent) => sum + (agent.confidence || 0), 0) / agentResponses.length;
      
      // 合并推荐建议，去重
      const allRecommendations = [];
      agentResponses.forEach(agent => {
        if (agent.recommendations && Array.isArray(agent.recommendations)) {
          allRecommendations.push(...agent.recommendations);
        }
      });
      
      // 去重
      const seen = new Set();
      const uniqueRecommendations = allRecommendations.filter(rec => {
        if (seen.has(rec)) return false;
        seen.add(rec);
        return true;
      });
      
      // 生成共识内容
      let content = "# 多智能体医疗团队共识意见\n\n";
      content += "基于我们多学科团队的综合分析：\n\n";
      
      // 添加各专家的核心观点
      agentResponses.forEach(agent => {
        if (agent.content) {
          const firstLine = agent.content.split('\n')[0] || '';
          content += `**${agent.agent_name}观点**：${firstLine}\n\n`;
        }
      });
      
      // 添加综合建议
      content += "## 综合建议\n";
      uniqueRecommendations.slice(0, 7).forEach((rec, index) => {
        content += `${index + 1}. ${rec}\n`;
      });
      
      content += "\n*请注意：以上建议仅供参考，具体治疗方案请遵循您的主治医生的建议。*";
      
      return {
        content,
        confidence,
        recommendations: uniqueRecommendations.slice(0, 7)
      };
    };
    
    // 发送消息主函数 - 异步处理方式
    const sendMessage = async (message) => {
      if (!message.trim() || !isConnected.value) return;

      isLoading.value = true;
      
      try {
        // 添加用户消息到界面
        messages.value.push({
          id: Date.now(),
          type: "user",
          content: message.trim(),
          timestamp: new Date().toISOString(),
        });
        
        // 滚动到底部
        await nextTick();
        scrollToBottom();

        // 获取相关历史记录
        const relevantHistory = getRelevantHistory();
        
        const requestData = {
          message: message,
          conversation_history: relevantHistory,
        };
        
        // 发送异步请求
        console.log('发送异步聊天请求...');
        const asyncResponse = await apiClient.post('/chat/async', requestData);
        const { request_id, initial_response } = asyncResponse.data;
        
        console.log('收到异步响应，请求ID:', request_id);
        
        // 立即显示初步响应
        const initialMessageId = Date.now();
        messages.value.push({
          id: initialMessageId,
          type: "async_initial",
          content: initial_response,
          request_id: request_id,
          timestamp: new Date().toISOString(),
        });
        
        await nextTick();
        scrollToBottom();
        
        // 开始轮询获取进度和结果
        await pollChatStatus(request_id, initialMessageId);
        
      } catch (error) {
        console.error('异步请求失败:', error);
        
        // 添加错误消息
        messages.value.push({
          id: Date.now(),
          type: "error",
          content: error.response?.data?.detail || "发送请求失败，请稍后重试",
          timestamp: new Date().toISOString(),
        });
        
      } finally {
        isLoading.value = false;
        
        // 确保滚动到底部
        await nextTick();
        scrollToBottom();
      }
    };
    
    // 轮询聊天状态
    const pollChatStatus = async (requestId, initialMessageId) => {
      let pollInterval;
      let processedAgents = new Set(); // 跟踪已处理的智能体
      
      try {
        // 轮询函数
        const poll = async () => {
          try {
            const statusResponse = await apiClient.get(`/chat/status/${requestId}`);
            const { status, agent_responses, consensus } = statusResponse.data;
            
            console.log('轮询状态:', status, '已获取响应的智能体:', Object.keys(agent_responses).length);
            
            // 更新初步响应消息的状态
            const initialMessage = messages.value.find(msg => msg.id === initialMessageId);
            if (initialMessage) {
              initialMessage.status = status;
            }
            
            // 所有智能体类型及其显示名称
            const agentTypes = {
              "oncologist": "肿瘤科医生",
              "radiologist": "影像科医生",
              "nurse": "肿瘤专科护士",
              "patient_advocate": "患者权益倡导者",
              "psychologist": "肿瘤心理专家"
            };
            
            // 检查并显示新的智能体响应
            for (const [agentId, agentData] of Object.entries(agent_responses)) {
              if (!processedAgents.has(agentId)) {
                processedAgents.add(agentId);
                const agentName = agentTypes[agentId] || agentId;
                
                // 更新智能体状态
                updateAgentStatus(agentId, "completed", "分析完成");
                
                // 添加智能体回复
                messages.value.push({
                  id: Date.now(),
                  type: "agent",
                  agent: agentId,
                  agentName: agentName,
                  content: agentData.content || "无内容",
                  confidence: agentData.confidence || 0.5,
                  recommendations: agentData.recommendations || [],
                  timestamp: new Date().toISOString(),
                });
                
                await nextTick();
                scrollToBottom();
              }
            }
            
            // 如果处理完成且有共识结果
            if (status === "completed" && consensus) {
              // 更新推荐建议
              currentRecommendations.value = consensus.recommendations || [];
              
              // 添加共识消息
              messages.value.push({
                id: Date.now(),
                type: "consensus",
                content: consensus.content || "无法生成共识意见",
                confidence: consensus.confidence || 0.5,
                timestamp: new Date().toISOString(),
              });
              
              await nextTick();
              scrollToBottom();
              
              // 清除轮询
              clearInterval(pollInterval);
              
              // 重置智能体状态（延迟执行）
              setTimeout(() => {
                agents.forEach((agent) => {
                  updateAgentStatus(agent.id, "idle", "就绪");
                });
              }, 3000);
            }
            // 如果发生错误
            else if (status === "error") {
              messages.value.push({
                id: Date.now(),
                type: "error",
                content: "处理分析请求时发生错误，请稍后重试",
                timestamp: new Date().toISOString(),
              });
              
              clearInterval(pollInterval);
            }
            
          } catch (error) {
            console.error('轮询状态失败:', error);
            // 继续轮询，直到达到最大次数
          }
        };
        
        // 立即执行一次轮询
        await poll();
        
        // 设置轮询间隔（1.5秒轮询一次）
        pollInterval = setInterval(poll, 1500);
        
        // 设置最大轮询时间（5分钟）
        setTimeout(() => {
          if (pollInterval) {
            clearInterval(pollInterval);
            
            // 检查是否已完成
            const initialMessage = messages.value.find(msg => msg.id === initialMessageId);
            if (initialMessage && initialMessage.status !== "completed") {
              messages.value.push({
                id: Date.now(),
                type: "warning",
                content: "分析超时，请刷新页面后查看结果或重新提交请求",
                timestamp: new Date().toISOString(),
              });
              
              // 重置智能体状态
              agents.forEach((agent) => {
                updateAgentStatus(agent.id, "idle", "就绪");
              });
            }
          }
        }, 300000); // 5分钟 = 300000毫秒
        
      } catch (error) {
        console.error('轮询过程中发生错误:', error);
        if (pollInterval) {
          clearInterval(pollInterval);
        }
      }
    };

    const updateAgentStatus = (agentId, status, statusText) => {
      const agent = agents.find((a) => a.id === agentId);
      if (agent) {
        agent.status = status;
        agent.statusText = statusText;
      }
    };

    const scrollToBottom = () => {
      // 使用requestAnimationFrame确保在DOM更新后执行滚动
      requestAnimationFrame(() => {
        const container = document.querySelector(".messages-container");
        if (container) {
          container.scrollTop = container.scrollHeight;
        }
      });
    };

    onMounted(() => {
      // 立即进行一次健康检查
      checkHealth();

      // 设置定期健康检查（每10秒，减少频率避免频繁请求）
      healthCheckInterval.value = setInterval(checkHealth, 100000);
    });

    onUnmounted(() => {
      // 清除健康检查定时器
      if (healthCheckInterval.value) {
        clearInterval(healthCheckInterval.value);
      }
    });

    return {
      isConnected,
      isLoading,
      messages,
      agents,
      currentRecommendations,
      inputPlaceholder,
      sendMessage,
      scrollToBottom,
    };
  },
};
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.app-container {
  width: 100vw;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
}

.app-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 1rem 2rem;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1400px;
  margin: 0 auto;
}

.app-header h1 {
  color: #2c3e50;
  font-size: 1.5rem;
  font-weight: 600;
}

.header-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #666;
  font-size: 0.9rem;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #e74c3c;
  transition: background 0.3s;
}

.status-indicator.connected {
  background: #2ecc71;
  box-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
}

.main-content {
  flex: 1;
  display: grid;
  grid-template-columns: 280px 1fr 300px;
  gap: 1rem;
  padding: 1rem;
  max-width: 1400px;
  width: 100%;
  margin: 0 auto;
  overflow: hidden;
}

.agents-panel,
.recommendations-panel {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 1.5rem;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
  overflow-y: auto;
}

.agents-panel h3,
.recommendations-panel h3 {
  color: #2c3e50;
  margin-bottom: 1rem;
  font-size: 1.1rem;
}

.agent-card {
  background: #f8f9fa;
  border-radius: 10px;
  padding: 1rem;
  margin-bottom: 1rem;
  display: flex;
  gap: 1rem;
  transition: transform 0.2s, box-shadow 0.2s;
}

.agent-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.agent-avatar {
  font-size: 2rem;
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: white;
  border-radius: 10px;
}

.agent-info {
  flex: 1;
}

.agent-info h4 {
  color: #2c3e50;
  font-size: 1rem;
  margin-bottom: 0.25rem;
}

.agent-specialty {
  color: #7f8c8d;
  font-size: 0.85rem;
  margin-bottom: 0.5rem;
}

.agent-status {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
}

.agent-status.idle {
  background: #ecf0f1;
  color: #7f8c8d;
}

.agent-status.working {
  background: #fff3cd;
  color: #856404;
  animation: pulse 1.5s infinite;
}

.agent-status.completed {
  background: #d4edda;
  color: #155724;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.chat-container {
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.recommendations-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.recommendation-item {
  display: flex;
  gap: 0.75rem;
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 3px solid #667eea;
  transition: transform 0.2s;
}

.recommendation-item:hover {
  transform: translateX(5px);
}

.rec-number {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  background: #667eea;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  font-weight: bold;
}

.rec-text {
  color: #2c3e50;
  font-size: 0.9rem;
  line-height: 1.5;
}

.no-recommendations {
  text-align: center;
  padding: 2rem 1rem;
  color: #7f8c8d;
}

.no-recommendations p {
  margin-bottom: 0.5rem;
}

.hint {
  font-size: 0.85rem;
  color: #95a5a6;
}

@media (max-width: 1200px) {
  .main-content {
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr auto;
  }

  .agents-panel,
  .recommendations-panel {
    display: none;
  }
}
</style>
