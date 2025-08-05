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
    
    # Exclude the final classification head from LoRA
    if 'score' in lora_module_names:
        lora_module_names.remove('score')
    if 'classifier' in lora_module_names:
        lora_module_names.remove('classifier')
        
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
        
        label_tensor = torch.tensor(label_item, dtype=torch.float) if self.use_soft_labels else torch.tensor(label_item, dtype=torch.long)
            
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor,
            'hard_labels': torch.tensor(hard_label_item, dtype=torch.long)
        }

class BaseModel:
    """Class for models with quantization and LoRA fine-tuning, supporting both hard and soft labels."""
    
    def __init__(self, model_name, num_labels, lora_config_dict=None, tuning_method='lora'):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing BaseModel on device '{self.device}' with base model: {model_name}")
        logging.info(f"Using tuning method: {tuning_method.upper()}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left' # Required for some models with flash attention
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else '[PAD]'

        if tuning_method == 'qlora':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            logging.info("Loaded 4-bit quantization config for QLoRA.")
        elif tuning_method == 'lora':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logging.info("Loaded 8-bit quantization config for LoRA.")
        else:
            quantization_config = None
            logging.info("No quantization will be used.")

        try:
            logging.info("Attempting to load model with Flash Attention 2...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, quantization_config=quantization_config,
                torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                device_map="auto", trust_remote_code=True,
            )
            logging.info("Successfully loaded model with Flash Attention 2 enabled.")
        except Exception as e:
            logging.warning(f"Could not load model with Flash Attention 2 (Error: {e}). Falling back to standard attention.")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels,
                quantization_config=quantization_config,
                device_map="auto", trust_remote_code=True,
            )

        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        if quantization_config:
            model = prepare_model_for_kbit_training(model)
        
        if lora_config_dict:
            logging.info(f"Applying LoRA/QLoRA config: {lora_config_dict}")
            target_modules = find_all_linear_names(model)
            logging.info(f"Found target modules for LoRA: {target_modules}")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, r=lora_config_dict['r'],
                lora_alpha=lora_config_dict['lora_alpha'], lora_dropout=lora_config_dict['lora_dropout'],
                target_modules=target_modules, inference_mode=False,
            )
            self.model = get_peft_model(model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = model

    def prepare_data_loader(self, df, text_col, target_col, batch_size, max_length, prompt_template, use_soft_labels, original_hard_label_col):
        if df is None or df.empty:
            return None
        texts = df[text_col].tolist()
        labels_for_loss = df[target_col].tolist()
        original_hard_labels = df[original_hard_label_col].tolist()
        
        dataset = FlexibleLabelDataset(texts, labels_for_loss, self.tokenizer, max_length, prompt_template, use_soft_labels, original_hard_labels)
        
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, train_loader, val_loader, epochs, learning_rate, output_dir, use_soft_labels, loss_alpha=0.5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        if use_soft_labels:
            criterion_kl = nn.KLDivLoss(reduction='batchmean')
            criterion_ce_for_soft = nn.CrossEntropyLoss()
            activation_log_softmax = nn.LogSoftmax(dim=-1)
            logging.info("Training with SOFT labels using a dynamic hybrid loss (alpha={:.2f}).".format(loss_alpha))
        else:
            criterion_ce_for_hard = nn.CrossEntropyLoss()
            logging.info("Training with HARD labels using Cross Entropy Loss.")

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                training_target = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                
                if use_soft_labels:
                    hard_labels = batch['hard_labels'].to(self.device)
                    log_probs = activation_log_softmax(logits)
                    loss_kl = criterion_kl(log_probs, training_target)
                    loss_ce = criterion_ce_for_soft(logits, hard_labels)
                    loss = loss_alpha * loss_kl + (1 - loss_alpha) * loss_ce
                else:
                    loss = criterion_ce_for_hard(logits, training_target)

                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, f"Epoch {epoch+1} Validation")
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                logging.info(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = os.path.join(output_dir, "best_model_lora")
                    self.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    logging.info(f"  ---> New best model saved to {best_model_path} with Val Acc: {best_val_acc:.4f}")
        
        return history
    
    def evaluate(self, data_loader, description="Evaluating"):
        """Evaluates the model using hard labels for loss and accuracy."""
        self.model.eval()
        total_loss, all_preds, all_ground_truth = 0, [], []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=description, leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
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

    def predict(self, texts, max_length, batch_size, return_probabilities=False, prompt_template=None):
        self.model.eval()
        all_preds, all_confidences, all_probabilities = [], [], []
        
        if prompt_template:
            texts = [prompt_template.format(user_message=text) for text in texts]

        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors='pt', padding=True, 
                truncation=True, max_length=max_length
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                confidences, preds = torch.max(probabilities, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_preds, all_confidences, all_probabilities
    
    @classmethod
    def from_adapter(cls, base_model_name, num_labels, adapter_dir, tuning_method='lora'):
        """Loads a model from a saved LoRA adapter."""
        logging.info(f"Loading base model '{base_model_name}' for adapter merging...")
        instance = cls(base_model_name, num_labels, lora_config_dict=None, tuning_method=tuning_method)
        logging.info(f"Loading and merging LoRA adapter from {adapter_dir}...")
        instance.model = PeftModel.from_pretrained(instance.model.base_model, adapter_dir)
        instance.model = instance.model.merge_and_unload()
        logging.info("Adapter merged and unloaded successfully.")
        instance.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        return instance
