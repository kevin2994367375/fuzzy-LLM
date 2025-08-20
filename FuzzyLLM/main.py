import matplotlib
matplotlib.use('Agg')

import os
import argparse
import pandas as pd
from pathlib import Path
import logging
import yaml
import numpy as np
import torch
import json
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample

from src.data.data_loader import DataLoader, split_data, save_processed_data
from src.data.fuzzy_encoder import AdaptiveFuzzyEncoder
from src.models.model import BaseModel
from src.utils.visualization import (
    plot_training_history, plot_confusion_matrix, 
    plot_classification_report, plot_label_distribution
)

def stratified_sample_df(df, label_col, n_samples_per_class):
    """Performs stratified sampling on a dataframe to get a balanced subset for debugging."""
    if df is None or df.empty:
        logging.warning("Input DataFrame for sampling is empty. Returning None.")
        return None
    
    min_class_count = df[label_col].value_counts().min()
    n_to_sample = min(min_class_count, n_samples_per_class)
    
    if n_to_sample < n_samples_per_class:
        logging.warning(f"Requested {n_samples_per_class} samples, but smallest class has {min_class_count}. Sampling {n_to_sample} instead.")
    if n_to_sample == 0:
        logging.error(f"Cannot sample 0 items. Smallest class might be empty or too small.")
        return pd.DataFrame(columns=df.columns)

    return df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(n=n_to_sample, random_state=42))

def fuzzy_judge_analysis(class_names, test_labels, test_preds, test_confidences, output_dir):
    """Analyzes model predictions using a fuzzy logic approach on confidence scores."""
    logging.info("\n=== Running Fuzzy Judge Analysis on Model Confidence ===")
    analysis_results = []
    test_labels, test_preds = np.asarray(test_labels), np.asarray(test_preds)
    if test_labels.size == 0 or test_preds.size == 0:
        logging.warning("No test labels/predictions to analyze in fuzzy_judge_analysis.")
        return {}
        
    for i in range(len(test_labels)):
        true_label_name = class_names[test_labels[i]] if test_labels[i] < len(class_names) else "Unknown"
        pred_label_name = class_names[test_preds[i]] if test_preds[i] < len(class_names) else "Unknown"
        confidence = test_confidences[i]
        is_correct = (test_labels[i] == test_preds[i])

        if confidence > 0.9: confidence_level = "Very High"
        elif confidence > 0.7: confidence_level = "High"
        elif confidence > 0.5: confidence_level = "Medium"
        else: confidence_level = "Low"
            
        if is_correct: diagnosis = "Confident & Correct" if confidence > 0.85 else "Correct but Uncertain"
        else: diagnosis = "Confident but Wrong" if confidence > 0.7 else "Wrong & Uncertain"
            
        analysis_results.append({
            'True Label': true_label_name, 'Predicted Label': pred_label_name, 
            'Confidence': float(confidence), 'Is Correct': is_correct, 
            'Confidence Level': confidence_level, 'Diagnosis Category': diagnosis
        })
        
    df_analysis = pd.DataFrame(analysis_results)
    
    category_counts = df_analysis['Diagnosis Category'].value_counts(normalize=True)
    stats_to_return = {f"fuzzy_{cat.replace(' & ', '_and_').replace(' ', '_').lower()}_percent": perc * 100 for cat, perc in category_counts.items()}

    logging.info("--- Fuzzy Diagnosis Summary ---")
    for category, percentage in category_counts.items():
        count = int(percentage * len(df_analysis))
        logging.info(f"{category:<25}: {count} samples ({percentage * 100:.2f}%)")
        
    fuzzy_output_path = os.path.join(output_dir, "fuzzy_confidence_analysis.csv")
    df_analysis.to_csv(fuzzy_output_path, index=False)
    logging.info(f"Fuzzy confidence analysis saved to {fuzzy_output_path}")
    
    return stats_to_return
    
def balance_dataset(df, label_col):
    """
    Applies random undersampling to the majority class to balance the ENTIRE dataset.
    """
    if df is None or df.empty: return None
    class_counts = df[label_col].value_counts()
    if len(class_counts) < 2: return df

    majority_class_label = class_counts.idxmax()
    target_size = int(class_counts.iloc[1]) if len(class_counts) > 1 else class_counts.min()

    logging.info(f"Balancing full dataset. Original distribution:\n{class_counts.to_string()}")
    
    df_majority = df[df[label_col] == majority_class_label]
    df_minority = df[df[label_col] != majority_class_label]
    
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=target_size, random_state=42)
    
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    logging.info(f"Balancing complete. New distribution:\n{df_balanced[label_col].value_counts().to_string()}")
    
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

def parse_args():
    """Parse command line arguments for the multi-strategy experiment platform."""
    parser = argparse.ArgumentParser(description='Multi-Strategy & Prototype Fuzzy-LLM Diagnosis Platform')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the main configuration file.')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Base directory for raw datasets.')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Directory for processed data splits.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Root directory for all outputs.')
    parser.add_argument('--model_name', type=str, default='./models/Qwen2.5-3B-Instruct', help='Base pretrained model path.')
    parser.add_argument('--balance_dataset', action='store_true',
                        help="If set, applies random undersampling to the training set's majority class.")
    parser.add_argument('--tuning_method', type=str, default='lora', choices=['lora', 'qlora'],
                        help="Choose the fine-tuning method: 'lora' (8-bit) or 'qlora' (4-bit).")
    parser.add_argument('--strategy', type=str, default='fuzzy', 
                        choices=['numeric', 'fuzzy'], 
                        help="The main experiment strategy.")
    parser.add_argument('--use_soft_labels', action='store_true',
                        help="If set, uses KNN-generated soft labels for the training target. Otherwise, uses hard labels.")
    parser.add_argument('--fuzzy_config_name', type=str, default='quantile_based_5_levels', 
                        help="The name of the fuzzy MF generation config to use (for 'fuzzy_knn' strategy).")
    parser.add_argument('--knn_k', type=int, default=30, 
                        help="Number of neighbors (k) for KNN soft label generation (for 'fuzzy_knn' strategy).")
    parser.add_argument('--dataset', type=str, default=None, help='Override active_dataset in config (e.g., cwru or pu).')
    parser.add_argument('--criterion', type=str, default=None, choices=['location', 'severity'], help='Train for a specific task only.')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--max_length', type=int, default=640, help='Maximum sequence length.')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout.')
    parser.add_argument('--soft_label_temperature', type=float, default=0.5,
                        help="Temperature for soft label sharpening (T<1: sharper, T>1: smoother).")
    parser.add_argument('--sharpening_schedule', type=str, default='constant',
                        choices=['constant', 'progressive'],
                        help="Schedule for sharpening: constant or progressive.")
    parser.add_argument('--loss_alpha', type=float, default=0.25,
                        help="The alpha weight for the KL-divergence part of the hybrid loss (0 <= alpha <= 1).")
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help="Level of Gaussian noise to add to test set features. "
                             "Expressed as a fraction of the feature's standard deviation. "
                             "Default is 0.0 (no noise).")
    parser.add_argument('--debug_sample_size', type=int, default=None, 
                        help="If set, uniformly samples N items per class from the training set for quick debugging. E.g., 100.")
    return parser.parse_args()

def add_gaussian_noise(df, feature_cols, noise_level=0.0):
    """
    为给定的DataFrame的特征列添加高斯噪声。
    噪声的强度由noise_level控制，它代表噪声标准差相对于原始特征标准差的比例。
    """
    if df is None or df.empty or noise_level == 0.0:
        return df
    logging.info(f"Injecting Gaussian noise with level: {noise_level}")
    df_noisy = df.copy()
    for col in feature_cols:
        if col in df_noisy.columns and pd.api.types.is_numeric_dtype(df_noisy[col]):
            noise_std = df_noisy[col].std() * noise_level
            noise = np.random.normal(0, noise_std, df_noisy[col].shape)
            df_noisy[col] = df_noisy[col] + noise
    return df_noisy

def get_prompt_template(task, use_soft_labels, class_names, input_strategy):
    """
    Returns the final, refined, and contextualized prompt templates.
    This version separates the instruction from the data placeholder.
    """
    if not class_names: return ""
    class_options = ", ".join(class_names[:-1]) + f", or {class_names[-1]}" if len(class_names) > 1 else class_names[0]
    
    if input_strategy == 'fuzzy':
        instruction = (
            "Role setup: You are an experienced bearing fault analyst specializing in fuzzy diagnostics. "
            "You are provided with a linguistic summary of a bearing's vibration characteristics.\n\n"
            "Task description: Your task is to interpret this fuzzy feature analysis and "
            "determine the probability of different fault types. Please provide your diagnosis "
            f"as a probability distribution over the following possible conditions: {class_options}."
        )

    elif input_strategy == 'numeric':
        if use_soft_labels:
            instruction = (
                "Role setup: You are an experienced bearing fault analyst. You are provided with a set "
                "of key numerical features extracted from a bearing's vibration signal.\n\n"
                "Task description: Your task is to analyze these numerical features and determine the "
                "probability of different fault types. Please provide your diagnosis as a "
                f"probability distribution over the following possible conditions: {class_options}."
            )
        else: # Numeric-Hard
            instruction = (
                "Role setup: You are an experienced bearing fault analyst. You are provided with a set "
                "of key numerical features extracted from a bearing's vibration signal.\n\n"
                "Task description: Your task is to analyze these numerical features and classify the "
                "health state of the bearing. Please provide only the single most likely fault type "
                f"from the following options: {class_options}."
            )
            
    if input_strategy == 'fuzzy':
        data_template = "The analysis is as follows: '{user_message}'"
    else: # numeric
        data_template = "The feature set is as follows: '{user_message}'"

    prompt_template = f"{instruction}\n\n{data_template}"
    
    return prompt_template

def save_test_results_to_csv(test_df, predictions, probabilities, class_names, output_dir, label_col, text_col, filename_suffix=""):
    """
    Saves the detailed test results to a CSV file, now with a filename suffix.
    """
    logging.info(f"Saving detailed test results with suffix '{filename_suffix}'...")
    true_labels = test_df[label_col].tolist()
    true_label_names = [class_names[i] for i in true_labels]
    predicted_label_names = [class_names[i] for i in predictions]
    results_data = {
        'LLM_Input_Text': test_df[text_col].tolist(),
        'True_Label': true_label_names,
        'Predicted_Label': predicted_label_names,
    }
    for i, class_name in enumerate(class_names):
        results_data[f'Prob_{class_name.replace(" ", "_")}'] = [p[i] for p in probabilities]
    results_df = pd.DataFrame(results_data)
    output_filename = f"test_predictions_detailed{filename_suffix}.csv"
    output_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Detailed test results saved to {output_path}")

def generate_knn_soft_labels(train_df, target_df, feature_cols, label_col, k, num_classes, scaler):
    """Generates fuzzy soft labels based on KNN, using a pre-fitted scaler."""
    logging.info(f"Generating KNN-based soft labels for {len(target_df)} samples with k={k}...")
    train_features_scaled = scaler.transform(train_df[feature_cols])
    target_features_scaled = scaler.transform(target_df[feature_cols])
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(train_features_scaled)
    _, indices = knn.kneighbors(target_features_scaled)
    soft_labels = []
    for i in tqdm(range(len(target_df)), desc=f"Generating Soft Labels for {len(target_df)} samples"):
        neighbor_labels = train_df.iloc[indices[i]][label_col].values
        label_counts = Counter(neighbor_labels)
        soft_label_vector = np.array([label_counts.get(j, 0) for j in range(num_classes)]) / k
        soft_labels.append(soft_label_vector.tolist())
    return soft_labels

def sharpen_soft_labels(soft_labels, temperature=0.5):
    """Applies temperature scaling to soft labels for sharpening."""
    if temperature == 1.0: return soft_labels
    logging.info(f"Applying soft label sharpening with T={temperature:.2f}")
    scaled = np.array(soft_labels) / temperature
    return softmax(scaled, axis=1).tolist()

def train_and_evaluate_task(task, label_col, class_names, args, train_df, val_df, test_df):
    """
    A robust function that ONLY handles model training and evaluation on pre-processed, pre-split data.
    """
    use_soft_labels = args.use_soft_labels
    input_strategy_name = args.strategy
    output_strategy_name = "soft" if args.use_soft_labels else "hard"
    
    strategy_name = f"{input_strategy_name}_input_{output_strategy_name}_target"
    if args.use_soft_labels:
        strategy_name += f"_k{args.knn_k}_T{args.soft_label_temperature}_alpha{args.loss_alpha}"
    
    strategy_name = f"{args.tuning_method}_{strategy_name}"
    if args.balance_dataset: strategy_name += '_balanced'
    if args.debug_sample_size: strategy_name += f'_debug{args.debug_sample_size}'
    
    logging.info(f"\n==================== Starting Workflow for Task: {task.title()} | Strategy: {strategy_name.upper()} ====================")
    
    output_dir = os.path.join(args.output_dir, args.dataset, task, f"strategy_{strategy_name}")
    processed_dir = os.path.join(args.processed_dir, args.dataset, task, f"strategy_{strategy_name}")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(processed_dir, exist_ok=True)
    
    save_processed_data(train_df, val_df, test_df, processed_dir)
    plot_label_distribution(train_df[label_col], f'{args.dataset.upper()} ({strategy_name}) Dist.', class_names, output_dir)
    
    model = BaseModel(model_name=args.model_name, num_labels=len(class_names), lora_config_dict={'r': args.lora_r, 'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout}, tuning_method=args.tuning_method)
    
    prompt_template = get_prompt_template(task, use_soft_labels, class_names)
    
    text_col = 'llm_input_text'; target_col = 'llm_target_output'
    train_loader = model.prepare_data_loader(train_df, text_col, target_col, args.batch_size, args.max_length, prompt_template, use_soft_labels=use_soft_labels, original_hard_label_col=label_col)
    val_loader = model.prepare_data_loader(val_df, text_col, target_col, args.batch_size, args.max_length, prompt_template, use_soft_labels=use_soft_labels, original_hard_label_col=label_col)
    
    history = model.train(
        train_loader, 
        val_loader, 
        args.epochs, 
        args.learning_rate, 
        output_dir, 
        use_soft_labels=use_soft_labels, 
        loss_alpha=args.loss_alpha
    )
    plot_training_history(history, f"{args.dataset.upper()} - {task.title()} ({strategy_name})", output_dir)
    
    best_adapter_path = os.path.join(output_dir, 'best_model_lora'); logging.info(f"\n=== Final Evaluation on Test Set ===")
    if not os.path.exists(best_adapter_path):
        logging.error(f"Best model not found."); return

    eval_model = BaseModel.from_adapter(
        base_model_name=args.model_name, 
        num_labels=len(class_names), 
        adapter_dir=best_adapter_path,
        tuning_method=args.tuning_method
    )
    
    eval_prompt_template = get_prompt_template(task, use_soft_labels=use_soft_labels, class_names= class_names)
    test_texts = test_df[text_col].tolist()
    
    _ , _, test_probabilities = eval_model.predict(
        test_texts, args.max_length, args.batch_size, 
        return_probabilities=True, prompt_template=eval_prompt_template
    )

    test_preds = np.argmax(test_probabilities, axis=1)
    test_confidences = np.max(test_probabilities, axis=1)
    test_labels = test_df[label_col].to_numpy()
    
    final_accuracy = accuracy_score(test_labels, test_preds)
    logging.info(f"Final Test Accuracy (post-processed from generated output): {final_accuracy:.4f}")
    
    report_dict = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_string = classification_report(test_labels, test_preds, target_names=class_names, digits=4, zero_division=0)
    logging.info("--- Classification Report ---")
    logging.info("\n" + report_string)
    
    report_path = os.path.join(output_dir, "classification_report.txt")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Classification Report for Task: {task} ({strategy_name})\n")
            f.write("="*70 + "\n")
            f.write(report_string)
        logging.info(f"Classification report text saved to: {report_path}")
    except Exception as e:
        logging.error(f"Failed to save classification report text: {e}")
        
    plot_confusion_matrix(test_labels, test_preds, class_names, f"CM - {args.dataset.upper()} ({task}) ({strategy_name})", output_dir)
    plot_classification_report(test_labels, test_preds, class_names, f"Report - {args.dataset.upper()} ({task}) ({strategy_name})", output_dir)
    save_test_results_to_csv(test_df, test_preds, test_probabilities, class_names, output_dir, label_col, text_col)
    fuzzy_stats = fuzzy_judge_analysis(class_names, test_labels, test_preds, test_confidences, output_dir)
    
    results = {
        'dataset': args.dataset,
        'task': task,
        'strategy': strategy_name,
        'tuning_method': args.tuning_method,
        'loss_alpha': args.loss_alpha if use_soft_labels else 'N/A',
        'accuracy': final_accuracy,
        'macro_avg_precision': report_dict['macro avg']['precision'],
        'macro_avg_recall': report_dict['macro avg']['recall'],
        'macro_avg_f1-score': report_dict['macro avg']['f1-score'],
        'weighted_avg_precision': report_dict['weighted avg']['precision'],
        'weighted_avg_recall': report_dict['weighted avg']['recall'],
        'weighted_avg_f1-score': report_dict['macro avg']['f1-score'],
    }
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            results[f"f1_{class_name.replace(' ', '_')}"] = metrics['f1-score']
            
    if fuzzy_stats:
        results.update(fuzzy_stats)

    return results

def main():
    """Main function with a robust, task-isolated workflow."""
    args = parse_args()
    
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    dataset_name = args.dataset if args.dataset else config.get('active_dataset', 'cwru'); args.dataset = dataset_name
    
    summary_file_path = os.path.join(args.output_dir, f"{args.dataset}_ablation_summary.csv")
    all_results = []
    
    data_loader = DataLoader(args.data_dir, dataset_name=dataset_name)
    features_df = data_loader.load_and_extract_features()
    if features_df is None: return

    if dataset_name == 'cwru':
        all_tasks = [
            ('location', 'Fault_Type', ['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault']),
            ('severity', 'Fault_Severity', ['Slight', 'Moderate', 'Severe', 'Critical'])
        ]
    elif dataset_name == 'hust':
        all_tasks = [
            ('location', 'Fault_Type', ['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault'])
        ]
    elif args.dataset == 'pu':
        location_classes = ['Normal', 'Inner Race Fault', 'Outer Race Fault', 'Compound Fault']
        all_tasks = [
            ('location', 'Fault_Type', location_classes)
        ]
    else:
        logging.error(f"Task definition for dataset '{args.dataset}' is not implemented.")
        return
    
    tasks_to_run = [t for t in all_tasks if not args.criterion or t[0] == args.criterion]
    
    for task, label_col_str, master_class_names in tasks_to_run:
        logging.info(f"\n\n<<<<<<<<<< Processing and Running for Task: {task.upper()} >>>>>>>>>>")
        
        label_mapping = {name: i for i, name in enumerate(master_class_names)}
        label_col_numeric = f"{task}_numeric"
        task_df = features_df.copy(); task_df[label_col_numeric] = task_df[label_col_str].map(label_mapping).fillna(-1).astype(int)
        task_df = task_df[task_df[label_col_numeric] != -1].reset_index(drop=True)
        if task_df.empty: logging.warning(f"No samples for task '{task}'. Skipping."); continue
            
        active_class_names = [name for i, name in enumerate(master_class_names) if i in task_df[label_col_numeric].unique()]
        
        train_df, val_df, test_df = split_data(task_df, label_col=label_col_numeric)
        if train_df is None or train_df.empty: continue

        if args.debug_sample_size and args.debug_sample_size > 0:
            train_df = stratified_sample_df(train_df, label_col_numeric, args.debug_sample_size)
            if val_df is not None: val_df = stratified_sample_df(val_df, label_col_numeric, max(1, int(args.debug_sample_size * 0.2)))
            if test_df is not None: test_df = stratified_sample_df(test_df, label_col_numeric, max(1, int(args.debug_sample_size * 0.2)))
            if train_df is None: continue

        dataset_config = config['datasets'][dataset_name]
        op_condition_features = dataset_config.get('op_condition_features', [])
        signal_features = dataset_config.get('signal_features', [])
        
        if not op_condition_features and not signal_features:
            numeric_features_list = dataset_config.get('numeric_features', [])
            if not numeric_features_list:
                logging.error(f"No feature keys found for dataset '{dataset_name}' in config."); continue
        else:
            numeric_features_list = op_condition_features + signal_features

        logging.info(f"Using {len(numeric_features_list)} features for task '{task}'.")
        
        test_df = add_gaussian_noise(test_df, numeric_features_list, noise_level=args.noise_level)
        
        scaler = StandardScaler().fit(train_df[numeric_features_list])
        
        text_col = 'llm_input_text'; target_col = 'llm_target_output'

        if args.strategy == 'fuzzy':
            logging.info("Preparing 'fuzzy' text inputs...")
            mf_config = config['adaptive_fuzzy_encoder_configs'][args.fuzzy_config_name]
            encoder = AdaptiveFuzzyEncoder(numeric_df=train_df[numeric_features_list], labels=train_df[label_col_numeric], config=mf_config)
            for df_split in [train_df, val_df, test_df]:
                if df_split is not None:
                    df_split.loc[:, text_col] = encoder.batch_encode_to_text(df_split[numeric_features_list])
        
        elif args.strategy == 'numeric':
            logging.info("Preparing 'numeric' text inputs...")
            def row_to_scaled_numeric_text(row): return ", ".join([f"{col.split('_')[-1]}: {val:.4f}" for col, val in row.items()])
            for df_split in [train_df, val_df, test_df]:
                if df_split is not None:
                    scaled_features = scaler.transform(df_split[numeric_features_list])
                    scaled_features_df = pd.DataFrame(scaled_features, columns=numeric_features_list, index=df_split.index)
                    df_split.loc[:, text_col] = scaled_features_df.apply(row_to_scaled_numeric_text, axis=1)
        
        if args.use_soft_labels:
            logging.info("Preparing SOFT-LABEL targets...")
            num_classes = len(active_class_names)
            train_soft_labels = generate_knn_soft_labels(train_df, train_df, numeric_features_list, label_col_numeric, args.knn_k, num_classes, scaler)
            if val_df is not None:
                val_soft_labels = generate_knn_soft_labels(train_df, val_df, numeric_features_list, label_col_numeric, args.knn_k, num_classes, scaler)
            
            train_df.loc[:, target_col] = sharpen_soft_labels(train_soft_labels, args.soft_label_temperature)
            if val_df is not None:
                val_df.loc[:, target_col] = sharpen_soft_labels(val_soft_labels, args.soft_label_temperature)
        else:
            logging.info("Preparing HARD-LABEL targets...")
            train_df.loc[:, target_col] = train_df[label_col_numeric]
            if val_df is not None:
                val_df.loc[:, target_col] = val_df[label_col_numeric]
        
        task_results = train_and_evaluate_task(task, label_col_numeric, active_class_names, args, train_df, val_df, test_df)
        if task_results:
            all_results.append(task_results)
            
    if all_results:
        summary_df = pd.DataFrame(all_results)
        file_exists = os.path.isfile(summary_file_path)
        summary_df.to_csv(summary_file_path, mode='a', header=not file_exists, index=False)
        logging.info(f"\n=== Ablation results appended to {summary_file_path} ===")
        
        run_summary_string = summary_df.to_string()
        print("\n--- Run Summary ---")
        print(run_summary_string)
        
        summary_txt_path = summary_file_path.replace('.csv', '.txt')
        try:
            with open(summary_txt_path, 'a', encoding='utf-8') as f:
                f.write(f"--- Summary for run at {pd.Timestamp.now()} ---\n")
                f.write(run_summary_string)
                f.write("\n\n" + "="*80 + "\n\n")
            logging.info(f"Run summary text also saved to: {summary_txt_path}")
        except Exception as e:
            logging.error(f"Failed to save run summary text: {e}")
        
    logging.info("\n=== Workflow Complete! ===")

if __name__ == "__main__":
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger(); root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    console_handler = logging.StreamHandler(); console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    main()
