from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import torch

class MedicalPretrainer:
    """医疗领域预训练器"""
    
    def __init__(self, base_model: str = "Qwen/Qwen2.5-14B"):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def prepare_medical_corpus(self, corpus_files: List[str], max_length: int = 2048) -> Dataset:
        """准备医疗文本语料"""
        
        # 加载和合并所有文本文件
        all_texts = []
        for file_path in corpus_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
                all_texts.extend([text.strip() for text in texts if text.strip()])
        
        # 分词处理
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
        
        # 创建数据集
        dataset = Dataset.from_dict({'text': all_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        return tokenized_dataset
    
    def continue_pretrain(self, dataset: Dataset,
                         output_dir: str = "./qwen2.5-med-pretrain",
                         num_epochs: int = 1,
                         learning_rate: float = 1e-5,
                         batch_size: int = 8):
        """继续预训练"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_steps=1000,
            logging_steps=100,
            save_steps=5000,
            fp16=True,
            dataloader_pin_memory=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print("Starting continue pretraining...")
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Continue pretraining completed. Model saved to {output_dir}")