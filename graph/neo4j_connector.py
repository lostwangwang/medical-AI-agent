# graph/neo4j_connector.py
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Tuple
import json
import logging

class Neo4jConnector:
    """Neo4j图数据库连接器"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)
        
        # 验证连接
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info("Neo4j connection established")
        except Exception as e:
            self.logger.error(f"Neo4j connection failed: {e}")
            self.driver = None
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
    
    def create_medical_schema(self):
        """创建医疗知识图谱模式"""
        constraints_and_indexes = [
            # 创建约束
            "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT treatment_name IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (m:Drug) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
            
            # 创建索引
            "CREATE INDEX disease_category IF NOT EXISTS FOR (d:Disease) ON (d.category)",
            "CREATE INDEX treatment_type IF NOT EXISTS FOR (t:Treatment) ON (t.type)",
            "CREATE INDEX drug_target IF NOT EXISTS FOR (m:Drug) ON (m.target)",
        ]
        
        with self.driver.session() as session:
            for query in constraints_and_indexes:
                try:
                    session.run(query)
                    self.logger.info(f"Executed: {query}")
                except Exception as e:
                    self.logger.warning(f"Failed to execute {query}: {e}")
    
    def add_disease_node(self, disease_info: Dict[str, Any]) -> str:
        """添加疾病节点"""
        query = """
        MERGE (d:Disease {name: $name})
        SET d.category = $category,
            d.icd_code = $icd_code,
            d.description = $description,
            d.severity = $severity,
            d.prevalence = $prevalence
        RETURN d.name as name
        """
        
        with self.driver.session() as session:
            result = session.run(query, **disease_info)
            return result.single()["name"] if result.single() else None
    
    def add_symptom_relationships(self, disease_name: str, symptoms: List[Dict[str, Any]]):
        """添加疾病-症状关系"""
        for symptom in symptoms:
            query = """
            MATCH (d:Disease {name: $disease_name})
            MERGE (s:Symptom {name: $symptom_name})
            SET s.type = $symptom_type,
                s.description = $description
            MERGE (d)-[r:HAS_SYMPTOM]->(s)
            SET r.frequency = $frequency,
                r.severity = $severity
            """
            
            params = {
                'disease_name': disease_name,
                'symptom_name': symptom['name'],
                'symptom_type': symptom.get('type', ''),
                'description': symptom.get('description', ''),
                'frequency': symptom.get('frequency', 0.0),
                'severity': symptom.get('severity', 'unknown')
            }
            
            with self.driver.session() as session:
                session.run(query, params)
    
    def add_treatment_relationships(self, disease_name: str, treatments: List[Dict[str, Any]]):
        """添加疾病-治疗关系"""
        for treatment in treatments:
            query = """
            MATCH (d:Disease {name: $disease_name})
            MERGE (t:Treatment {name: $treatment_name})
            SET t.type = $treatment_type,
                t.description = $description,
                t.standard_of_care = $standard_of_care
            MERGE (d)-[r:TREATED_BY]->(t)
            SET r.efficacy = $efficacy,
                r.evidence_level = $evidence_level,
                r.line_of_therapy = $line_of_therapy
            """
            
            params = {
                'disease_name': disease_name,
                'treatment_name': treatment['name'],
                'treatment_type': treatment.get('type', ''),
                'description': treatment.get('description', ''),
                'standard_of_care': treatment.get('standard_of_care', False),
                'efficacy': treatment.get('efficacy', 0.0),
                'evidence_level': treatment.get('evidence_level', ''),
                'line_of_therapy': treatment.get('line_of_therapy', 1)
            }
            
            with self.driver.session() as session:
                session.run(query, params)
    
    def add_drug_relationships(self, treatment_name: str, drugs: List[Dict[str, Any]]):
        """添加治疗-药物关系"""
        for drug in drugs:
            query = """
            MATCH (t:Treatment {name: $treatment_name})
            MERGE (m:Drug {name: $drug_name})
            SET m.generic_name = $generic_name,
                m.drug_class = $drug_class,
                m.mechanism = $mechanism,
                m.target = $target
            MERGE (t)-[r:INCLUDES_DRUG]->(m)
            SET r.dosage = $dosage,
                r.frequency = $frequency,
                r.duration = $duration
            """
            
            params = {
                'treatment_name': treatment_name,
                'drug_name': drug['name'],
                'generic_name': drug.get('generic_name', ''),
                'drug_class': drug.get('drug_class', ''),
                'mechanism': drug.get('mechanism', ''),
                'target': drug.get('target', ''),
                'dosage': drug.get('dosage', ''),
                'frequency': drug.get('frequency', ''),
                'duration': drug.get('duration', '')
            }
            
            with self.driver.session() as session:
                session.run(query, params)
    
    def find_treatment_paths(self, disease_name: str, patient_characteristics: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """查找治疗路径"""
        query = """
        MATCH (d:Disease {name: $disease_name})-[r:TREATED_BY]->(t:Treatment)
        OPTIONAL MATCH (t)-[drug_rel:INCLUDES_DRUG]->(m:Drug)
        RETURN d.name as disease,
               t.name as treatment,
               t.type as treatment_type,
               r.efficacy as efficacy,
               r.evidence_level as evidence_level,
               r.line_of_therapy as line_of_therapy,
               collect({drug: m.name, dosage: drug_rel.dosage, mechanism: m.mechanism}) as drugs
        ORDER BY r.line_of_therapy ASC, r.efficacy DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, disease_name=disease_name)
            return [record.data() for record in result]
    
    def find_similar_cases(self, symptoms: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """基于症状查找相似病例"""
        query = """
        MATCH (s:Symptom)<-[:HAS_SYMPTOM]-(d:Disease)
        WHERE s.name IN $symptoms
        WITH d, count(s) as symptom_overlap
        MATCH (d)-[:TREATED_BY]->(t:Treatment)
        RETURN d.name as disease,
               d.category as category,
               symptom_overlap,
               collect(t.name) as treatments
        ORDER BY symptom_overlap DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, symptoms=symptoms, limit=limit)
            return [record.data() for record in result]
    
    def get_drug_interactions(self, drug_names: List[str]) -> List[Dict[str, Any]]:
        """查询药物相互作用"""
        query = """
        MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
        WHERE d1.name IN $drug_names AND d2.name IN $drug_names
        RETURN d1.name as drug1,
               d2.name as drug2,
               r.interaction_type as interaction_type,
               r.severity as severity,
               r.description as description
        """
        
        with self.driver.session() as session:
            result = session.run(query, drug_names=drug_names)
            return [record.data() for record in result]