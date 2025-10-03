import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
from typing import List, Dict, Any

class MedicalSFTTrainer:
    """医疗SFT微调训练器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, training_data: List[Dict[str, Any]], max_length: int = 2048) -> Dataset:
        """准备训练数据集"""
        
        def tokenize_function(examples):
            # 组合instruction和output
            inputs = []
            for instruction, output in zip(examples['instruction'], examples['output']):
                # 使用特殊格式
                full_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                inputs.append(full_text)
            
            # 分词
            model_inputs = self.tokenizer(
                inputs,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            # 设置labels（用于计算loss）
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        # 创建Dataset
        dataset = Dataset.from_list(training_data)
        
        # 应用分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, 
              output_dir: str = "./qwen2.5-med-sft",
              num_epochs: int = 3,
              learning_rate: float = 2e-5,
              batch_size: int = 4,
              gradient_accumulation_steps: int = 4):
        """执行SFT训练"""
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # 禁用wandb等
        )
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        print("Starting SFT training...")
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"SFT training completed. Model saved to {output_dir}")