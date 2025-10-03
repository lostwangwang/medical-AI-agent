import json
import os
from typing import List, Dict, Any, Optional
from .vector_store import FAISSVectorStore, WeaviateVectorStore

class MedicalKnowledgeBase:
    """医疗知识库"""
    
    def __init__(self, vector_store_type: str = "faiss", **kwargs):
        # 初始化向量存储
        if vector_store_type.lower() == "faiss":
            self.vector_store = FAISSVectorStore(**kwargs)
        elif vector_store_type.lower() == "weaviate":
            self.vector_store = WeaviateVectorStore(**kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
        
        # 知识源配置
        self.knowledge_sources = {
            'clinical_guidelines': [],  # 临床指南
            'research_papers': [],      # 研究论文
            'case_studies': [],         # 病例研究
            'drug_information': [],     # 药物信息
            'medical_protocols': []     # 医疗协议
        }
    
    def load_clinical_guidelines(self, guidelines_path: str) -> None:
        """加载临床指南"""
        if not os.path.exists(guidelines_path):
            print(f"Guidelines path not found: {guidelines_path}")
            return
        
        documents = []
        
        # 示例：NCCN肿瘤治疗指南
        nccn_guidelines = {
            'breast_cancer': {
                'title': 'NCCN乳腺癌临床实践指南',
                'content': '''
                乳腺癌诊疗指南要点：
                
                1. 诊断流程：
                - 体格检查和病史采集
                - 影像学检查（钼靶、超声、MRI）
                - 组织病理学确诊
                - 免疫组化检测（ER、PR、HER2）
                - 基因检测（BRCA1/2、21基因）
                
                2. 分期系统：
                - TNM分期标准
                - 临床分期和病理分期
                - 分子分型分类
                
                3. 治疗策略：
                - 手术治疗：保乳手术vs乳房切除术
                - 新辅助化疗适应症
                - 辅助化疗方案选择
                - 内分泌治疗指征
                - 靶向治疗（曲妥珠单抗、CDK4/6抑制剂）
                - 放疗指征和技术
                
                4. 随访监测：
                - 定期体格检查
                - 影像学复查频率
                - 肿瘤标志物监测
                - 晚期并发症管理
                ''',
                'category': 'clinical_guidelines',
                'source': 'NCCN Guidelines v2024',
                'cancer_type': 'breast',
                'version': '2024.1'
            },
            'lung_cancer': {
                'title': 'NCCN肺癌临床实践指南', 
                'content': '''
                肺癌诊疗指南要点：
                
                1. 筛查和早期发现：
                - 高危人群筛查策略
                - 低剂量CT筛查指征
                - 肺结节管理流程
                
                2. 诊断和分期：
                - 组织学分类（小细胞vs非小细胞）
                - 基因检测面板（EGFR、ALK、ROS1、KRAS、PD-L1）
                - TNM第8版分期系统
                - 纵隔淋巴结评估
                
                3. 非小细胞肺癌治疗：
                - 早期NSCLC手术治疗
                - 辅助化疗指征
                - 局部晚期NSCLC同步放化疗
                - 驱动基因阳性靶向治疗
                - 免疫检查点抑制剂应用
                
                4. 小细胞肺癌治疗：
                - 局限期SCLC治疗方案
                - 广泛期SCLC系统治疗
                - 脑转移预防和治疗
                ''',
                'category': 'clinical_guidelines',
                'source': 'NCCN Guidelines v2024',
                'cancer_type': 'lung',
                'version': '2024.1'
            }
        }
        
        for cancer_type, guideline in nccn_guidelines.items():
            documents.append(guideline)
            self.knowledge_sources['clinical_guidelines'].append(guideline)
        
        # 添加到向量存储
        self.vector_store.add_documents(documents)
        print(f"Loaded {len(documents)} clinical guidelines")
    
    def load_research_papers(self, papers_path: str) -> None:
        """加载研究论文"""
        # 示例研究论文数据
        research_papers = [
            {
                'title': 'CAR-T细胞治疗在血液肿瘤中的最新进展',
                'content': '''
                CAR-T细胞治疗代表了癌症免疫治疗的重大突破。本综述总结了CAR-T技术的最新发展：
                
                1. 技术原理：
                - 嵌合抗原受体（CAR）设计
                - T细胞基因改造技术
                - 细胞扩增和回输流程
                
                2. 临床应用：
                - CD19 CAR-T治疗B细胞恶性肿瘤
                - BCMA CAR-T治疗多发性骨髓瘤
                - 实体瘤CAR-T治疗探索
                
                3. 疗效数据：
                - 急性淋巴细胞白血病缓解率90%+
                - 弥漫性大B细胞淋巴瘤缓解率80%+
                - 多发性骨髓瘤缓解率73%
                
                4. 副作用管理：
                - 细胞因子释放综合征(CRS)
                - 免疫效应细胞相关神经毒性综合征(ICANS)
                - 预防和处理策略
                ''',
                'authors': ['张三', '李四', '王五'],
                'journal': 'Nature Medicine',
                'year': 2024,
                'doi': '10.1038/nm.2024.001',
                'category': 'research_papers',
                'source': 'PubMed',
                'keywords': ['CAR-T', '免疫治疗', '血液肿瘤']
            },
            {
                'title': '人工智能在肿瘤影像诊断中的应用与前景',
                'content': '''
                人工智能特别是深度学习在医学影像诊断中展现出巨大潜力：
                
                1. 技术发展：
                - 卷积神经网络在影像识别中的应用
                - 多模态影像融合技术
                - 联邦学习在医疗数据中的应用
                
                2. 临床应用现状：
                - 肺结节检测和分类
                - 乳腺癌钼靶筛查
                - 皮肤癌病变识别
                - 视网膜病变诊断
                
                3. 性能评估：
                - 肺癌检测敏感性94.4%
                - 乳腺癌筛查准确率89.9%
                - 皮肤癌诊断准确率91.0%
                
                4. 挑战与机遇：
                - 数据质量和标准化
                - 算法可解释性
                - 监管审批路径
                - 医生接受度和培训
                ''',
                'authors': ['陈六', '赵七', '孙八'],
                'journal': 'The Lancet Digital Health',
                'year': 2024,
                'doi': '10.1016/S2589-7500(24)00001-X',
                'category': 'research_papers',
                'source': 'PubMed',
                'keywords': ['人工智能', '医学影像', '深度学习', '肿瘤诊断']
            }
        ]
        
        self.knowledge_sources['research_papers'].extend(research_papers)
        self.vector_store.add_documents(research_papers)
        print(f"Loaded {len(research_papers)} research papers")
    
    def load_drug_information(self, drugs_path: str) -> None:
        """加载药物信息"""
        drug_info = [
            {
                'title': '曲妥珠单抗（赫赛汀）药物信息',
                'content': '''
                通用名：曲妥珠单抗
                商品名：赫赛汀（Herceptin）
                
                1. 药物分类：
                - 单克隆抗体
                - HER2靶向治疗药物
                - 抗肿瘤药物
                
                2. 适应症：
                - HER2阳性转移性乳腺癌
                - HER2阳性早期乳腺癌辅助治疗
                - HER2阳性转移性胃癌
                
                3. 用法用量：
                - 负荷剂量：4mg/kg静脉输注
                - 维持剂量：2mg/kg，每周一次
                - 或8mg/kg负荷剂量，然后6mg/kg每3周一次
                
                4. 主要副作用：
                - 心脏毒性（左室射血分数下降）
                - 输注反应
                - 肺毒性
                - 感染风险增加
                
                5. 注意事项：
                - 治疗前评估心功能
                - 定期监测心脏功能
                - 避免与蒽环类药物同时使用
                ''',
                'drug_name': '曲妥珠单抗',
                'category': 'drug_information',
                'source': 'FDA药物标签',
                'atc_code': 'L01XC03',
                'target': 'HER2'
            },
            {
                'title': 'PD-1抑制剂帕博利珠单抗（可瑞达）',
                'content': '''
                通用名：帕博利珠单抗
                商品名：可瑞达（Keytruda）
                
                1. 药物分类：
                - 程序性死亡受体-1（PD-1）抑制剂
                - 免疫检查点抑制剂
                - 单克隆抗体
                
                2. 适应症：
                - 黑色素瘤
                - 非小细胞肺癌
                - 头颈部鳞癌
                - 霍奇金淋巴瘤
                - 尿路上皮癌
                - 微卫星不稳定性高(MSI-H)肿瘤
                
                3. 用法用量：
                - 成人：200mg静脉输注，每3周一次
                - 或400mg静脉输注，每6周一次
                - 输注时间：30分钟
                
                4. 免疫相关副作用：
                - 免疫相关肺炎
                - 免疫相关结肠炎
                - 免疫相关肝炎
                - 免疫相关内分泌病变
                - 免疫相关皮肤反应
                
                5. 管理要点：
                - 治疗前评估自身免疫病史
                - 密切监测免疫相关副作用
                - 及时使用皮质激素治疗严重副作用
                ''',
                'drug_name': '帕博利珠单抗',
                'category': 'drug_information', 
                'source': 'FDA药物标签',
                'atc_code': 'L01XC18',
                'target': 'PD-1'
            }
        ]
        
        self.knowledge_sources['drug_information'].extend(drug_info)
        self.vector_store.add_documents(drug_info)
        print(f"Loaded {len(drug_info)} drug information entries")
    
    def search_knowledge(self, query: str, category: Optional[str] = None, 
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索知识库"""
        # 基本向量搜索
        results = self.vector_store.search(query, top_k=top_k * 2)  # 获取更多结果用于过滤
        
        # 如果指定了类别，则过滤结果
        if category:
            results = [r for r in results if r.get('category') == category]
        
        # 返回前top_k个结果
        return results[:top_k]
    
    def get_knowledge_summary(self) -> Dict[str, int]:
        """获取知识库摘要统计"""
        summary = {}
        for category, items in self.knowledge_sources.items():
            summary[category] = len(items)
        return summary