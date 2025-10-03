# Medical AI Agent 项目目录结构

## 项目概述
本项目是一个基于多智能体的医疗AI系统，旨在为医疗场景提供智能化决策支持。

## 目录结构

```
medical-AI-agent/
├── agents/                          # 各类医疗智能体实现
│   ├── base_agent.py               # 智能体基类
│   ├── nurse.py                    # 护理智能体
│   ├── oncologist.py               # 肿瘤科医生智能体
│   ├── patient_advocate.py         # 患者权益代表智能体
│   ├── psychologist.py             # 心理学家智能体
│   └── radiologist.py              # 放射科医生智能体
├── backend/                        # 后端服务
│   ├── api/                        # API接口
│   │   └── chat.py                # 聊天相关接口
│   ├── models/                     # 数据模型
│   │   └── schemas.py             # 数据结构定义
│   ├── main.py                    # 后端主入口
│   └── requirements.txt           # 后端依赖
├── bin\bash/                       # 脚本文件
│   ├── deploy.sh                  # 部署脚本
│   └── package-lock.json          # npm依赖锁定文件
├── consensus/                      # 多智能体共识机制
│   ├── consensus_matrix.py        # 共识矩阵
│   └── decision_engine.py         # 决策引擎
├── data/                           # 数据文件
│   └── test_cases.json            # 测试用例数据
├── experiments/                    # 实验相关
│   ├── evaluator.py               # 评估器
│   └── experiment_runner.py       # 实验运行器
├── frontend/                       # 前端界面
│   ├── src/                        # 前端源码
│   │   ├── components/            # 组件
│   │   │   ├── ChatWindow.vue     # 聊天窗口组件
│   │   │   ├── InputBox.vue       # 输入框组件
│   │   │   └── MessageBubble.vue  # 消息气泡组件
│   │   ├── App.vue                # 主应用组件
│   │   ├── main.js                # 前端入口文件
│   │   └── style.css              # 样式文件
│   ├── index.html                 # 主页面
│   ├── package-lock.json          # npm依赖锁定文件
│   ├── package.json               # npm配置文件
│   └── vite.config.js             # Vite配置文件
├── graph/                          # 图数据库相关
│   ├── graph_rag.py               # 图RAG实现
│   └── neo4j_connector.py         # Neo4j连接器
├── rag/                            # 检索增强生成
│   ├── knowledge_base.py          # 知识库
│   ├── retriever.py               # 检索器
│   └── vector_store.py            # 向量存储
├── training/                       # 训练相关
│   ├── data_processor.py          # 数据处理器
│   ├── dpo_trainer.py             # DPO训练器
│   ├── pretrain.py                # 预训练
│   └── sft_trainer.py             # SFT训练器
├── README.md                       # 项目说明文档
├── debug.py                        # 调试工具
├── experiment_config.json          # 实验配置
├── main.py                         # 项目主入口
└── requirements.txt                # 项目依赖
```

## 技术栈

- 前端: Vue.js + Vite
- 后端: Python FastAPI
- 数据库: Neo4j 图数据库
- 通信: WebSocket
- 部署: Shell 脚本
- AI框架: PyTorch/TensorFlow (推测)

## 主要组件说明

1. **agents/** - 实现了不同医疗领域的专业AI智能体
2. **backend/** - 提供Web API服务和数据模型定义
3. **consensus/** - 处理多个智能体之间的决策共识
4. **frontend/** - 用户交互界面
5. **graph/** - 图数据库相关操作，用于知识图谱
6. **rag/** - 检索增强生成模块，用于知识检索
7. **training/** - 模型训练相关工具

## 使用方式
1. 项目中`.env`未配置时，请先配置。
> [!WARNING]
> 提交代码时，不要提交该文件!!!
```shell
# .env
QWEN_API_KEY=xxxx

# Neo4j数据库URI (默认: bolt://localhost:7687)
NEO4J_URI=bolt://localhost:7687

# Neo4j用户名 (默认: neo4j)
NEO4J_USER=neo4j

# Neo4j密码 (默认: password)
NEO4J_PASSWORD=your_secure_password_here
```
2. neo4j 
# neo4j
## neo4j 登录网址
http://localhost:7474/browser/

## 运行系统
1. 演示模式（推荐入门）:
```bash
python main.py --mode demo
```
2. 交互模式:
```bash
# 目前支持命令行交互，可以先使用该模式
python main.py --mode interactive
```
3. 完整实验:
```bash
python main.py --mode experiment --config experiment_config.json
```

## 后端代码运行

```shell
# 保证后端修改代码之后可以热重载
uvicorn backend.main:app --host localhost --port 8000 --reload
```

## 前端运行
```shell
# 进入前端目录

cd frontend
# 安装依赖
npm install

# 启动前端
npm run dev