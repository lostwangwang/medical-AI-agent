from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch

class MedicalDPOTrainer:
    """医疗DPO训练器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 创建参考模型（冻结的原始模型）
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def prepare_preference_dataset(self, preference_data: List[Dict[str, Any]]) -> Dataset:
        """准备偏好数据集"""
        
        def format_example(example):
            return {
                'prompt': example['prompt'],
                'chosen': example['chosen'],
                'rejected': example['rejected']
            }
        
        formatted_data = [format_example(ex) for ex in preference_data]
        return Dataset.from_list(formatted_data)
    
    def train(self, preference_dataset: Dataset,
              output_dir: str = "./qwen2.5-med-dpo",
              num_epochs: int = 1,
              learning_rate: float = 5e-7,
              batch_size: int = 2):
        """执行DPO训练"""
        
        # DPO配置
        dpo_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            beta=0.1,  # DPO温度参数
            warmup_steps=50,
            logging_steps=10,
            save_steps=500,
            fp16=True,
            remove_unused_columns=False,
            report_to=None,
        )
        
        # 创建DPO训练器
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            config=dpo_config,
            train_dataset=preference_dataset,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        print("Starting DPO training...")
        dpo_trainer.train()
        
        # 保存模型
        dpo_trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"DPO training completed. Model saved to {output_dir}")