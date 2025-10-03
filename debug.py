#!/usr/bin/env python3
# debug.py - 详细调试Pydantic错误
import sys
import traceback

def test_import(obj_name, import_func):
    print(f"测试: {obj_name}")
    try:
        result = import_func()
        print(f"✓ 成功: {obj_name}")
        return True
    except Exception as e:
        print(f"✗ 失败: {obj_name}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 开始详细导入测试 ===")
    
    # 1. 测试基础数据类
    print("\n" + "="*40)
    print("测试基础数据类")
    print("="*40)
    
    test_import("MedicalCase 导入", lambda: __import__('agents.base_agent').base_agent.MedicalCase)
    test_import("AgentOpinion 导入", lambda: __import__('agents.base_agent').base_agent.AgentOpinion)
    
    # 2. 测试ConsensusResult类
    print("\n" + "="*40)
    print("测试ConsensusResult类")
    print("="*40)
    
    test_import("ConsensusResult 导入", lambda: __import__('consensus.consensus_matrix').consensus_matrix.ConsensusResult)
    
    # 3. 测试Pydantic模型
    print("\n" + "="*40)
    print("测试Pydantic模型")
    print("="*40)
    
    test_import("Message 导入", lambda: __import__('backend.models.schemas').models.schemas.Message)
    test_import("AgentResponse 导入", lambda: __import__('backend.models.schemas').models.schemas.AgentResponse)
    test_import("ConsensusResult Pydantic 导入", lambda: __import__('backend.models.schemas').models.schemas.ConsensusResult)
    
    # 4. 测试向量存储类
    print("\n" + "="*40)
    print("测试向量存储类")
    print("="*40)
    
    test_import("VectorStore 导入", lambda: __import__('rag.vector_store').vector_store.VectorStore)
    test_import("FAISSVectorStore 导入", lambda: __import__('rag.vector_store').vector_store.FAISSVectorStore)
    
    print("\n=== 测试完成 ===")