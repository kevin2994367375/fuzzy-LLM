import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import logging
import scipy.io
import re
import pywt
from sklearn.preprocessing import StandardScaler
import scipy.stats

class DataLoader:
    def __init__(self, data_dir):
        """
        Initialize data loader with directory containing raw data files.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.raw_data = None
        
        # 只保留信号数据、故障严重性和故障类型
        self.text_column = 'Signal_Data'  # Column for signal features
        self.label_column = 'Fault_Severity'   # Only keep fault severity
        self.position_column = 'Fault_Type'  # Fault location
        
        # 故障严重性标签映射（英文）
        self.label_mapping = {
            'Critical': 0,
            'Severe': 1,
            'Moderate': 2,
            'Slight': 3
        }
        
        self.num_classes = 4
        
    def load_cwru_data(self, pattern="*.mat"):
        """
        Load CWRU bearing fault dataset .mat files, keep only signal, fault severity, and fault type. Apply wavelet transform and normalization. All labels in English. Wavelet features are per-level statistics.
        """
        all_data = []
        fault_type_map = {
            'normal': 'Normal',
            'B': 'Ball Fault',
            'IR': 'Inner Race Fault',
            'OR': 'Outer Race Fault'
        }
        severity_map = {
            '007': 'Slight',
            '014': 'Moderate',
            '021': 'Severe',
            '028': 'Critical'
        }
        severity_label_map = {
            'Critical': 'Critical',
            'Severe': 'Severe',
            'Moderate': 'Moderate',
            'Slight': 'Slight'
        }
        def extract_wavelet_stats(signal, wavelet='db4', level=3):
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            stats = []
            stats_names = []
            for i, c in enumerate(coeffs):
                stats.extend([
                    np.mean(c),
                    np.std(c),
                    np.sum(np.square(c)),
                    np.max(c),
                    np.min(c),
                    scipy.stats.kurtosis(c),
                    scipy.stats.skew(c)
                ])
                stats_names.extend([
                    f'Level{i}_Mean', f'Level{i}_Std', f'Level{i}_Energy', f'Level{i}_Max', f'Level{i}_Min', f'Level{i}_Kurtosis', f'Level{i}_Skewness'
                ])
            return np.array(stats), stats_names
        feature_list = []
        sample_list = []
        stats_names_ref = None
        for fault_type_dir in self.data_dir.glob("*"):
            if not fault_type_dir.is_dir():
                continue
            fault_type = fault_type_dir.name
            if fault_type == 'Normal' or fault_type.lower() == 'normal':
                fault_type_label = 'normal'
            else:
                fault_type_label = fault_type
            if fault_type_label == 'normal':
                for file_path in fault_type_dir.glob(pattern):
                    try:
                        file_name = file_path.name
                        mat_data = scipy.io.loadmat(file_path)
                        signal_key = None
                        for key in mat_data.keys():
                            if '_DE_time' in key or 'DE_time' in key:
                                signal_key = key
                                break
                        if signal_key is None:
                            for key in mat_data.keys():
                                if isinstance(mat_data[key], np.ndarray) and mat_data[key].size > 100:
                                    signal_key = key
                                    break
                        if signal_key is None:
                            logging.warning(f"Cannot find signal data in {file_name}")
                            continue
                        signal_data = mat_data[signal_key].flatten()
                        segment_length = 1024
                        for i in range(0, len(signal_data) - segment_length, segment_length * 10):
                            segment = signal_data[i:i+segment_length]
                            wavelet_feature, stats_names = extract_wavelet_stats(segment)
                            if stats_names_ref is None:
                                stats_names_ref = stats_names
                            stats_desc = "; ".join([f"{name}: {value:.6f}" for name, value in zip(stats_names, wavelet_feature)])
                            stats_desc = f"Wavelet statistical features: {stats_desc}"
                            sample = {
                                'Signal_Data': wavelet_feature,
                                'Signal_Stats_Description': stats_desc,
                                'Fault_Type': 'Normal',
                                'Fault_Severity': '',
                                'source_file': f"{fault_type}/{file_name}"
                            }
                            feature_list.append(wavelet_feature)
                            sample_list.append(sample)
                    except Exception as e:
                        logging.warning(f"Error loading {fault_type}/{file_name}: {e}")
                continue
            for fault_size_dir in fault_type_dir.glob("*"):
                if not fault_size_dir.is_dir():
                    continue
                fault_size = fault_size_dir.name
                severity = severity_map.get(fault_size, 'Moderate')
                for file_path in fault_size_dir.glob(pattern):
                    try:
                        file_name = file_path.name
                        mat_data = scipy.io.loadmat(file_path)
                        signal_key = None
                        for key in mat_data.keys():
                            if '_DE_time' in key or 'DE_time' in key:
                                signal_key = key
                                break
                        if signal_key is None:
                            for key in mat_data.keys():
                                if isinstance(mat_data[key], np.ndarray) and mat_data[key].size > 100:
                                    signal_key = key
                                    break
                        if signal_key is None:
                            logging.warning(f"Cannot find signal data in {file_name}")
                            continue
                        signal_data = mat_data[signal_key].flatten()
                        segment_length = 1024
                        for i in range(0, len(signal_data) - segment_length, segment_length // 2):
                            segment = signal_data[i:i+segment_length]
                            wavelet_feature, stats_names = extract_wavelet_stats(segment)
                            if stats_names_ref is None:
                                stats_names_ref = stats_names
                            stats_desc = "; ".join([f"{name}: {value:.6f}" for name, value in zip(stats_names, wavelet_feature)])
                            stats_desc = f"Wavelet statistical features: {stats_desc}"
                            sample = {
                                'Signal_Data': wavelet_feature,
                                'Signal_Stats_Description': stats_desc,
                                'Fault_Type': fault_type_map.get(fault_type_label, 'Unknown'),
                                'Fault_Severity': severity_label_map[severity],
                                'source_file': f"{fault_type}/{fault_size}/{file_name}"
                            }
                            feature_list.append(wavelet_feature)
                            sample_list.append(sample)
                    except Exception as e:
                        logging.warning(f"Error loading {fault_type}/{fault_size}/{file_name}: {e}")
        # Normalization
        if feature_list:
            scaler = StandardScaler()
            features_norm = scaler.fit_transform(np.array(feature_list))
            for idx, sample in enumerate(sample_list):
                sample['Signal_Data'] = features_norm[idx]
            self.raw_data = pd.DataFrame(sample_list)
            return self.raw_data
        else:
            logging.warning("No data found")
            return None
    
    # 保留原有的load_excel_files方法
    def load_excel_files(self, pattern="*.xlsx"):
        """
        Load all Excel files matching the pattern in the data directory.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            Dictionary of DataFrames with filenames as keys
        """
        data_dict = {}
        for file_path in self.data_dir.glob(pattern):
            try:
                df = pd.read_excel(file_path)
                data_dict[file_path.name] = df
                logging.info(f"Loaded {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                logging.warning(f"Error loading {file_path.name}: {e}")
        
        return data_dict
    
    def merge_dataframes(self, data_dict):
        """
        Merge multiple dataframes into a single one with source information.
        
        Args:
            data_dict: Dictionary of DataFrames
            
        Returns:
            Merged DataFrame
        """
        merged_data = []
        
        for file_name, df in data_dict.items():
            df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
            df['source_file'] = file_name
            merged_data.append(df)
            
        if merged_data:
            self.raw_data = pd.concat(merged_data, ignore_index=True)
            return self.raw_data
        else:
            logging.warning("No data to merge")
            return None
    
    def preprocess_data(self, df=None):
        """
        Process both fault severity and fault type, generate numeric labels (English labels)
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data available. Load data first.")
            df = self.raw_data.copy()
        else:
            df = df.copy()
        keep_cols = ['Signal_Data', 'Fault_Type', 'Fault_Severity', 'Signal_Stats_Description', 'source_file']
        df = df[[col for col in keep_cols if col in df.columns]]
        df[self.text_column] = df[self.text_column].astype(str).str.strip()
        df = df.dropna(subset=[self.text_column])
        # Fault severity to numeric (English labels)
        df['Fault_Severity_numeric'] = df['Fault_Severity'].apply(lambda x: self.label_mapping.get(x, -1) if pd.notna(x) else -1)
        # Fault type to numeric
        type_mapping = {
            'Inner': 0,
            'Outer': 1,
            'Ball': 2,
            'Normal': 3
        }
        df['Fault_Type_numeric'] = df['Fault_Type'].apply(lambda x: type_mapping.get(x, -1) if pd.notna(x) else -1)
        # 过滤掉无效标签
        unmapped_severity = (df['Fault_Severity_numeric'] == -1).sum()
        unmapped_type = (df['Fault_Type_numeric'] == -1).sum()
        if unmapped_severity > 0:
            logging.warning(f"Warning: {unmapped_severity} rows with unmapped labels in column 'Fault_Severity' will be dropped")
            df = df[df['Fault_Severity_numeric'] != -1]
        if unmapped_type > 0:
            logging.warning(f"Warning: {unmapped_type} rows with unmapped labels in column 'Fault_Type' will be dropped")
            df = df[df['Fault_Type_numeric'] != -1]
        return df
    
    def split_data(self, df, label_col='Fault_Severity_numeric', test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data by specified label column (numeric)
        """
        from sklearn.model_selection import train_test_split
        if label_col not in df.columns:
            raise ValueError(f"Numeric label column '{label_col}' not found. Run preprocess_data first.")
        df = df.dropna(subset=[label_col])
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=random_state, stratify=None)
        logging.info(f"Train set: {train_df.shape[0]} rows")
        logging.info(f"Validation set: {val_df.shape[0]} rows")
        logging.info(f"Test set: {test_df.shape[0]} rows")
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir, criterion=None):
        """
        Save processed data splits to disk.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            output_dir: Directory to save processed data
            criterion: Optional criterion name to include in filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add criterion to filenames if specified
        prefix = f"{criterion}_" if criterion else ""
        
        train_df.to_csv(output_dir / f"{prefix}train.csv", index=False)
        val_df.to_csv(output_dir / f"{prefix}val.csv", index=False)
        test_df.to_csv(output_dir / f"{prefix}test.csv", index=False)
        
        logging.info(f"Saved processed data to {output_dir}")
    
    def get_text_column(self):
        return self.text_column
    
    def get_label_column(self):
        return 'Fault_Severity_numeric'
    
    def get_num_classes(self):
        return self.num_classes
    
    def merge_and_report_conflicts(self, data_dict, conflict_output_path='conflicts_output.xlsx'):
        """
        Merge multiple dataframes, picking the most frequent value for label columns when grouped by key columns.
        Also, output rows where the label columns have conflicting values for the same group.
        
        Args:
            data_dict: Dictionary of DataFrames
            conflict_output_path: Path to save the conflicts Excel file
        Returns:
            merged_df: DataFrame with resolved values
            conflicts_df: DataFrame with conflicts
        """
        merged_data = []
        for file_name, df in data_dict.items():
            df = df.copy()
            df['source_file'] = file_name
            merged_data.append(df)
        if not merged_data:
            logging.warning("No data to merge")
            return None, None
        df = pd.concat(merged_data, ignore_index=True)
        # Group and aggregate by mode
        def mode_or_first(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else x.iloc[0]
        agg_dict = {col: mode_or_first for col in self.label_columns.values()}
        merged_df = df.groupby(['ConversationID', 'Case', 'JailbreakID', 'Conversation_Pair', 'User Message', 'Assistant Message'], as_index=False).agg(agg_dict)
        # Find conflicts
        def has_conflict(subdf):
            conflicts = {}
            for col in self.label_columns.values():
                vals = subdf[col].dropna().unique()
                if len(vals) > 1:
                    conflicts[col] = list(vals)
            return conflicts if conflicts else None
        conflict_rows = []
        for _, subdf in df.groupby(['ConversationID', 'Case', 'JailbreakID', 'Conversation_Pair', 'User Message', 'Assistant Message']):
            conflicts = has_conflict(subdf)
            if conflicts:
                row = {col: subdf.iloc[0][col] for col in ['ConversationID', 'Case', 'JailbreakID', 'Conversation_Pair', 'User Message', 'Assistant Message']}
                for col in self.label_columns.values():
                    row[col] = list(subdf[col].dropna().unique())
                conflict_rows.append(row)
        conflicts_df = pd.DataFrame(conflict_rows)
        if not conflicts_df.empty:
            conflicts_df.to_excel(conflict_output_path, index=False)
            logging.info(f"Conflicts saved to {conflict_output_path}")
        return merged_df, conflicts_df