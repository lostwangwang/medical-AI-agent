#!/bin/bash
# deploy.sh - 一键部署脚本

echo "🚀 开始部署医疗AI多智能体对话系统"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到Python3，请先安装Python"
    exit 1
fi

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 未找到Node.js，请先安装Node.js"
    exit 1
fi

# 启动后端
#echo "📦 安装后端依赖..."
#cd backend
#pip install -r requirements.txt

echo "🔧 启动后端服务..."
python main.py &
BACKEND_PID=$!
echo "后端PID: $BACKEND_PID"

# 等待后端启动
sleep 3

# 启动前端
echo "📦 安装前端依赖..."
cd ../frontend
npm install

echo "🎨 启动前端服务..."
npm run dev &
FRONTEND_PID=$!
echo "前端PID: $FRONTEND_PID"

echo "✅ 系统启动完成！"
echo "后端地址: http://localhost:8000"
echo "前端地址: http://localhost:5173"
echo "API文档: http://localhost:8000/docs"

# 等待用户中断
echo "按 Ctrl+C 停止所有服务"
wait

# 杀死服务
echo “lsof -i:{端口号}”
echo "kill -9 {PID}"