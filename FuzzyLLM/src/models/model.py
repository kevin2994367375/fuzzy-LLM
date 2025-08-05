import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
import logging
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
import bitsandbytes as bnb

def find_all_linear_names(model):
    """Dynamically finds all linear layers for LoRA injection."""
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'score' in lora_module_names: lora_module_names.remove('score')
    if 'classifier' in lora_module_names: lora_module_names.remove('classifier')
    return sorted(list(lora_module_names))

class FlexibleLabelDataset(Dataset):
    """
    Dataset that handles text inputs, and BOTH soft and hard labels simultaneously
    for robust training and evaluation.
    """
    def __init__(self, texts, labels_for_loss, tokenizer, max_length, prompt_template, use_soft_labels, original_hard_labels):
        self.texts = texts
        self.labels_for_loss = labels_for_loss
        self.original_hard_labels = original_hard_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.use_soft_labels = use_soft_labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_item = self.labels_for_loss[idx]
        hard_label_item = self.original_hard_labels[idx]
        
        if self.prompt_template:
            text = self.prompt_template.format(user_message=text)

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
        # The 'labels' key will hold the target for loss calculation
        label_tensor = torch.tensor(label_item, dtype=torch.float) if self.use_soft_labels else torch.tensor(label_item, dtype=torch.long)
            
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor,
            'hard_labels': torch.tensor(hard_label_item, dtype=torch.long)
        }

class BaseModel:
    """Class for models with 8-bit quantization and LoRA fine-tuning, supporting both hard and soft labels."""
    
    def __init__(self, model_name, num_labels, lora_config_dict=None, tuning_method='qlora'):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing BaseModel with base model: {model_name}")
        logging.info(f"Using tuning method: {tuning_method.upper()}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left' 
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token: self.tokenizer.pad_token = self.tokenizer.eos_token
            else: self.tokenizer.pad_token = self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})['pad_token']

        # 【【【 核心修改 2：根据 tuning_method 动态选择量化配置 】】】
        if tuning_method == 'qlora':
            # QLoRA uses 4-bit quantization with specific settings
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logging.info("Loaded 4-bit quantization config for QLoRA.")
        elif tuning_method == 'lora':
            # Standard LoRA can use 8-bit quantization
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logging.info("Loaded 8-bit quantization config for LoRA.")
        else:
            quantization_config = None # No quantization
            logging.info("No quantization will be used.")

        try:
            logging.info("Attempting to load model with Flash Attention 2...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16, # Flash Attention 2 works best with bfloat16 or float16
                attn_implementation="flash_attention_2", # This is the magic key!
                device_map="auto",
                trust_remote_code=True,
            )
            logging.info("Successfully loaded model with Flash Attention 2 enabled.")
        except Exception as e:
            # Fallback mechanism if flash_attn is not installed, not supported, or causes any error
            logging.warning(f"Could not load model with Flash Attention 2 (Error: {e}). Falling back to standard attention.")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )

        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # This is needed for both LoRA and QLoRA
        model = prepare_model_for_kbit_training(model)
        
        if lora_config_dict:
            logging.info(f"Applying LoRA/QLoRA config: {lora_config_dict}")
            target_modules = find_all_linear_names(model)
            logging.info(f"Found target modules: {target_modules}")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=lora_config_dict.get('r', 16),
                lora_alpha=lora_config_dict.get('lora_alpha', 32),
                lora_dropout=lora_config_dict.get('lora_dropout', 0.1),
                target_modules=target_modules
            )
            self.model = get_peft_model(model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = model

    def prepare_data_loader(self, df, text_col, target_col, batch_size, max_length, prompt_template, use_soft_labels, original_hard_label_col):
        texts = df[text_col].tolist()
        labels_for_loss = df[target_col].tolist()
        original_hard_labels = df[original_hard_label_col].tolist()
        
        dataset = FlexibleLabelDataset(texts, labels_for_loss, self.tokenizer, max_length, prompt_template, use_soft_labels, original_hard_labels)
        
        df.attrs['name'] = df.attrs.get('name', 'train')
        is_train = "train" in str(df.attrs['name']).lower()
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    
    def train(self, train_loader, val_loader, epochs, learning_rate, output_dir, use_soft_labels, loss_alpha=0.5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # 为软标签策略预定义损失函数和激活函数
        if use_soft_labels:
            criterion_kl = nn.KLDivLoss(reduction='batchmean')
            criterion_ce_for_soft = nn.CrossEntropyLoss() # 也为软标签的硬部分准备CE
            activation_log_softmax = nn.LogSoftmax(dim=-1)
            logging.info("Training with SOFT labels using a dynamic hybrid loss.")
        criterion_ce_for_hard = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                # 'labels' 键现在只用于软标签策略的目标
                soft_target = batch['labels'].to(self.device) 
                # 'hard_labels' 键始终是我们计算交叉熵损失的黄金标准
                hard_target = batch['hard_labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                
                # --- 根据策略计算损失 ---
                if use_soft_labels:
                    hard_labels = batch['hard_labels'].to(self.device)
                    log_probs = activation_log_softmax(logits)
                    
                     # 【【【 核心修改：使用传入的 loss_alpha 参数 】】】
                    alpha = loss_alpha # 使用从命令行传入的值
                    # KL散度使用软标签目标 (soft_target)
                    loss_kl = criterion_kl(log_probs, soft_target) 
                    # 交叉熵使用硬标签目标 (hard_target)
                    loss_ce = criterion_ce_for_hard(logits, hard_target) 
                    loss = alpha * loss_kl + (1 - alpha) * loss_ce
                else: # hard label training
                    loss = criterion_ce_for_hard(logits, hard_target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            if val_loader is not None:
                # 验证阶段只关心最终的分类性能，所以可以统一用CE Loss和Accuracy
                val_loss, val_acc = self.evaluate(val_loader, f"Epoch {epoch+1} Validation")
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logging.info(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                # 模型保存始终基于验证集准确率
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = os.path.join(output_dir, "best_model_lora")
                    self.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    logging.info(f"  ---> New best model found! Val Acc: {best_val_acc:.4f}. Saved to {best_model_path}")
            
            # 学习率调度器现在基于验证集损失
            if val_loader is not None:
                scheduler.step(val_loss)
        
        return history
    
    # 【【【 核心修改：简化 evaluate 方法，使其只负责硬标签评估 】】】
    def evaluate(self, data_loader, description="Evaluating"):
        """
        Evaluates the model. ALWAYS calculates loss and accuracy against hard labels.
        """
        self.model.eval()
        total_loss = 0
        all_preds, all_ground_truth = [], []
        
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=description, leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                # We ALWAYS use the 'hard_labels' for evaluation metrics
                hard_labels = batch['hard_labels'].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                
                loss = criterion(logits, hard_labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_ground_truth.extend(hard_labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_ground_truth, all_preds)
        
        return avg_loss, accuracy

    
    def predict(self, texts, max_length, batch_size, return_confidence=False, return_probabilities=False, prompt_template=None):
        self.model.eval()
        self.model.to(self.device)

        if prompt_template:
            formatted_texts = [prompt_template.format(user_message=text) for text in texts]
        else:
            formatted_texts = texts

        all_preds, all_confidences, all_probabilities = [], [], []
        
        progress_bar = tqdm(range(0, len(formatted_texts), batch_size), desc="Predicting")
        for i in progress_bar:
            batch_texts = formatted_texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors='pt', 
                padding=True, truncation=True, max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                logits = outputs.logits
                
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                confidences, preds = torch.max(probabilities, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_preds, all_confidences, all_probabilities
    
    @classmethod
    def from_adapter(cls, base_model_name, num_labels, adapter_dir, tuning_method='qlora'):
        """Loads a model from a saved LoRA adapter, correctly re-applying quantization and Flash Attention."""
        logging.info(f"Loading model from adapter with tuning method '{tuning_method}'...")
        # The __init__ method now handles all the setup, including Flash Attention.
        # We just need to initialize a new instance with the correct settings.
        model_instance = cls(
            model_name=base_model_name, 
            num_labels=num_labels, 
            lora_config_dict=None, 
            tuning_method=tuning_method
        )
        
        logging.info(f"Loading LoRA adapter from {adapter_dir} into the new base model...")
        model_instance.model = PeftModel.from_pretrained(model_instance.model, adapter_dir)
        # It's usually better to load the tokenizer from the adapter dir as well
        model_instance.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        logging.info("Adapter loaded successfully.")
        
        return model_instance