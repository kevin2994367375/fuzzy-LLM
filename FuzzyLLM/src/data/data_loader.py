# src/data/data_loader.py

import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.io
import pywt
from sklearn.model_selection import train_test_split
import scipy.stats
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from tqdm import tqdm
from pyentrp import entropy as ent
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class DataLoader:
    """
    Its ONLY responsibility is to load raw files and extract features into a DataFrame
    with human-readable string labels (e.g., 'Inner Race Fault').
    It does NOT handle any numeric encoding of labels.
    """
    def __init__(self, data_dir, dataset_name='cwru'):
        if data_dir:
            self.raw_data_dir = Path(data_dir)
            # Correctly handle path joining for different dataset names
            self.data_path = self.raw_data_dir / dataset_name.upper()
            if dataset_name.lower() == 'xjtu':
                 self.data_path = self.raw_data_dir / 'XJTU' # Specific folder name for XJTU
        else:
            self.raw_data_dir = None
            self.data_path = None

        self.dataset_name = dataset_name.lower()
        self._load_dataset_parameters()
        if self.data_path: logging.info(f"DataLoader configured for {self.dataset_name.upper()}.")
        
    def _load_dataset_parameters(self):
        """Loads physical parameters, including a detailed map for the PU dataset."""
        dataset_name = self.dataset_name
        if dataset_name == 'cwru':
            self.fs = 12000; self.rpm_map = {'0': 1797, '1': 1772, '2': 1750, '3': 1730}; self.default_rpm = 1797
            self.n_balls = 9; self.ball_dia = 0.3126 * 25.4 * 1e-3; self.pitch_dia = 1.516 * 25.4 * 1e-3; self.contact_angle = np.deg2rad(0)
        elif dataset_name in ['xjtu']:
            self.fs = 25600; self.n_balls = 8; self.ball_dia = 7.92 * 1e-3; self.pitch_dia = 39.7 * 1e-3; self.contact_angle = np.deg2rad(0)
            self.xjtu_fault_location_map = { ... }
        elif dataset_name == 'hust':
            self.fs = 25600; self.n_balls = 9; self.ball_dia = 7.94 * 1e-3; self.pitch_dia = 1.516 * 25.4 * 1e-3; self.contact_angle = np.deg2rad(0)
        elif dataset_name == 'pu':
            logging.info("Loading VERIFIED parameters for PU dataset (6203 Bearing).")
            self.fs = 64000
            self.n_balls, self.ball_dia, self.pitch_dia, self.contact_angle = 8, 9.525*1e-3, 28.5*1e-3, 0
            if self.data_path: logging.info(f"DataLoader configured for Paderborn (PU) dataset. Path: {self.data_path}")


    def load_and_extract_features(self):
        """Public dispatcher method to load data and extract features."""
        if self.data_path is None or not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path is not valid or was not provided: {self.data_path}")
        logging.info(f"Loading data and extracting features from: {self.data_path}")
        if self.dataset_name == 'cwru': return self._process_cwru_data()
        elif self.dataset_name == 'xjtu': return self._process_xjtu_data()
        elif self.dataset_name == 'pu': return self._process_pu_data()
        elif self.dataset_name == 'hust': return self._process_hust_data()

    def _process_hust_data(self):
        """
        Processes HUST data, correctly parsing files and fusing features from
        all three (X, Y, Z) vibration channels by calling the existing _extract_all_features_as_dict.
        """
        all_samples_info = []
        location_map = {'C': 'Normal', 'I': 'Inner Race Fault', 'O': 'Outer Race Fault', 'B': 'Ball Fault'}
        severity_map = {'0.5X': 'Slight', 'X': 'Severe'}
        
        xls_files = list(self.data_path.rglob("*.xls*"))
        if not xls_files:
            logging.error(f"No .xls or .xlsx files found in {self.data_path}."); return None

        for file_path in tqdm(xls_files, desc="Processing HUST Bearings"):
            try:
                filename = file_path.stem
                
                # --- 1. Filename Parsing (This logic is correct) ---
                parts = filename.split('_')
                if len(parts) < 2: continue
                fault_location, fault_severity, op_speed_hz, op_load_val = None, 'N/A', 0.0, 0.0
                if len(parts) == 3 and parts[1] in location_map:
                    fault_severity=severity_map.get(parts[0], 'Severe'); fault_location=location_map.get(parts[1])
                    op_speed_hz=float(re.search(r'(\d+)', parts[2]).group(1))
                    op_load_val = float(re.search(r'(\d+\.?\d*)', parts[0]).group(1)) if 'X' in parts[0] else 0
                elif len(parts) == 2 and parts[0] in location_map:
                    fault_location=location_map.get(parts[0])
                    if fault_location != 'Normal': fault_severity = 'Severe'
                    op_speed_hz = float(re.search(r'(\d+)', parts[1]).group(1))
                elif 'VS' in parts: continue
                else: continue
                if fault_location is None: continue

                # --- 2. Precise Excel Data Reading (This logic is correct) ---
                df_signals = pd.read_csv(file_path, sep=r'\s+', header=None, skiprows=22, on_bad_lines='skip')
                if df_signals.shape[1] < 5: continue
                signal_data_x = df_signals.iloc[:, 2].astype(np.float64).values
                signal_data_y = df_signals.iloc[:, 3].astype(np.float64).values
                signal_data_z = df_signals.iloc[:, 4].astype(np.float64).values

                # --- 3. Sliding Window and Feature Fusion ---
                segment_length, step = 4096, 2048
                min_len = min(len(signal_data_x), len(signal_data_y), len(signal_data_z))
                
                for j in range(0, min_len - segment_length, step):
                    
                    # 【【【 核心修正：正确地调用你已有的 _extract_all_features_as_dict 函数 】】】
                    # 它已经有了prefix参数，我们直接使用它
                    features_x = self._extract_all_features_as_dict(signal_data_x[j:j+segment_length], 0,0,0, prefix='x_')
                    features_y = self._extract_all_features_as_dict(signal_data_y[j:j+segment_length], 0,0,0, prefix='y_')
                    features_z = self._extract_all_features_as_dict(signal_data_z[j:j+segment_length], 0,0,0, prefix='z_')
                    
                    op_features = {'op_speed': op_speed_hz, 'op_load': op_load_val}
                    
                    combined_features = {**features_x, **features_y, **features_z, **op_features}
                    
                    sample = {'Fault_Type': fault_location, 'Fault_Severity': fault_severity,
                              'source_file': file_path.name, 'bearing_id': filename, **combined_features}
                    all_samples_info.append(sample)
            except Exception as e:
                logging.error(f"Error processing HUST file {file_path}: {e}")

        return pd.DataFrame(all_samples_info) if all_samples_info else None

    def _process_xjtu_data(self):
        """
        Processes XJTU-SY data using a sophisticated Health Indicator (HI) approach.
        """
        all_samples_info = []
        bearing_dirs = sorted([d for d in self.data_path.rglob("Bearing*") if d.is_dir()])
        
        for bearing_dir in tqdm(bearing_dirs, desc="Processing XJTU Bearings"):
            bearing_name = bearing_dir.name
            true_fault_type = self.xjtu_fault_location_map.get(bearing_name, 'Unknown')
            if 'Compound' in true_fault_type:
                logging.info(f"Skipping compound fault bearing: {bearing_name}"); continue

            csv_files = sorted(bearing_dir.glob("*.csv"), key=lambda p: int(re.search(r'(\d+)\.csv$', str(p)).group(1)))
            if not csv_files: continue
            
            # --- 1. Extract time-domain features for HI construction ---
            logging.info(f"Extracting trend features for {bearing_name} to build HI...")
            hi_features_list = []
            for f in csv_files:
                try:
                    signal = pd.read_csv(f, header=0).iloc[:, 0].astype(np.float64).values
                    hi_features_list.append(self._extract_hi_features(signal))
                except Exception:
                    # Append a row of NaNs if a file is corrupted
                    hi_features_list.append([np.nan] * 7) 
            
            hi_features_df = pd.DataFrame(hi_features_list, columns=['rms', 'kurtosis', 'skew', 'peak', 'p2p', 'crest', 'impulse'])
            hi_features_df.ffill(inplace=True) # Fill any NaNs with the previous value

            # --- 2. Build Health Indicator (HI) using PCA ---
            # Scale features before PCA
            scaler = MinMaxScaler()
            hi_features_scaled = scaler.fit_transform(hi_features_df)
            
            pca = PCA(n_components=1)
            health_index = pca.fit_transform(hi_features_scaled).flatten()
            
            # Ensure the HI represents degradation (higher value = worse health)
            # If correlation with time is negative, flip the HI curve
            if np.corrcoef(health_index, np.arange(len(health_index)))[0, 1] < 0:
                health_index = -health_index

            # Smooth the HI curve for more stable thresholding
            smoothed_hi = pd.Series(health_index).rolling(window=10, min_periods=1, center=True).mean().values

            # --- 3. Scientific Health Stage Division based on HI ---
            healthy_hi_mean = np.mean(smoothed_hi[:int(len(smoothed_hi) * 0.15)])
            healthy_hi_std = np.std(smoothed_hi[:int(len(smoothed_hi) * 0.15)])
            threshold = healthy_hi_mean + 3 * healthy_hi_std # 3-sigma rule on the smooth HI
            
            degradation_start_index = next((i for i, hi_val in enumerate(smoothed_hi) if hi_val > threshold), len(csv_files))
            
            degradation_duration = len(csv_files) - degradation_start_index
            early_fault_end_index = degradation_start_index + int(degradation_duration * 0.4) # Give more room for early stage

            logging.info(f"Bearing {bearing_name}: HI-based degradation detected at file index {degradation_start_index}.")

            # --- 4. Final Feature Extraction and Label Assignment ---
            for i, file_path in enumerate(csv_files):
                try:
                    signal_data = pd.read_csv(file_path, header=0).iloc[:, 0].astype(np.float64).values
                    
                    if i < degradation_start_index: fault_type = 'Normal'
                    elif i < early_fault_end_index: fault_type = 'Early_Fault'
                    else: fault_type = true_fault_type
                    
                    # We still need op_conditions from path
                    try:
                        condition_dir = file_path.parent.parent.name
                        match = re.match(r'(\d+\.?\d*)Hz(\d+\.?\d*)kN', condition_dir)
                        op_speed_hz, op_load_kn = float(match.group(1)), float(match.group(2))
                    except Exception: op_speed_hz, op_load_kn = 35.0, 12.0

                    segment_length, step = 4096, 2048
                    for j in range(0, len(signal_data) - segment_length, step):
                        segment = signal_data[j:j+segment_length]
                        features = self._extract_all_features_as_dict(segment, bpfi=0, bpfo=0, bsf=0)
                        features['op_speed'] = op_speed_hz
                        features['op_load'] = op_load_kn
                        sample = {'Fault_Type': fault_type, 'source_file': f"{file_path.name}_seg{j//step}", 'bearing_id': f"{condition_dir}_{bearing_name}", **features}
                        all_samples_info.append(sample)
                except Exception as e:
                    logging.error(f"Error in final processing for {file_path}: {e}")

        return pd.DataFrame(all_samples_info) if all_samples_info else None

    # 【【【 新增：专门用于HI构建的特征提取函数 】】】
    def _extract_hi_features(self, signal):
        """Extracts a set of time-domain features known to be sensitive to degradation."""
        rms = np.sqrt(np.mean(signal**2))
        kurtosis = scipy.stats.kurtosis(signal)
        skewness = scipy.stats.skew(signal)
        peak = np.max(np.abs(signal))
        peak_to_peak = np.max(signal) - np.min(signal)
        crest_factor = peak / rms if rms > 0 else 0
        impulse_factor = peak / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
        return np.array([rms, kurtosis, skewness, peak, peak_to_peak, crest_factor, impulse_factor])
    

    # ... (All other private helper methods like _process_cwru_data, _get_feature_names, etc., remain unchanged) ...
    def _process_cwru_data(self):
        all_samples_info = []; fault_type_map = {'B': 'Ball Fault', 'IR': 'Inner Race Fault', 'OR': 'Outer Race Fault', 'Normal': 'Normal'}
        severity_map = {'007': 'Slight', '014': 'Moderate', '021': 'Severe', '028': 'Critical'}
        for file_info in tqdm(self._get_cwru_file_list(), desc="Extracting Features from CWRU"):
            try:
                file_path, fault_type_key, severity_key = file_info['path'], file_info['fault_type_key'], file_info['severity_key']
                match = re.search(r'_(\d)\.mat$', file_path.name); rpm = self.rpm_map.get(match.group(1) if match else '0', self.default_rpm)
                fault_freqs = self._calculate_fault_frequencies(rpm); mat_data = scipy.io.loadmat(file_path); signal_key = next((k for k in mat_data if 'DE_time' in k), None)
                if not signal_key: continue
                signal_data = mat_data[signal_key].flatten(); segment_length, step = 4096 , 2048
                for i in range(0, len(signal_data) - segment_length, step):
                    segment = signal_data[i:i+segment_length]; features = self._extract_all_features_as_dict(segment, **fault_freqs)
                    sample = {'Fault_Type': fault_type_map[fault_type_key], 'Fault_Severity': severity_map.get(severity_key, 'N/A'), 'source_file': file_path.name, **features}
                    all_samples_info.append(sample)
            except Exception as e: logging.error(f"Error processing CWRU file {file_path}: {e}")
        return pd.DataFrame(all_samples_info) if all_samples_info else None

    def _process_pu_data(self):
        """Processes all PU files and returns a single DataFrame of features."""
        all_samples_info = []
        folder_to_label_map = {"Healthy": "Normal", "Outer Ring Fault": "Outer Race Fault", 
                               "Inner Ring Fault": "Inner Race Fault", "Compound Fault": "Compound Fault"}
        
        all_files_to_process = []
        for fault_dir in self.data_path.glob("*"):
            if fault_dir.is_dir() and fault_dir.name in folder_to_label_map:
                for file_path in fault_dir.glob("**/*.mat"):
                    all_files_to_process.append({'path': file_path, 'type': folder_to_label_map[fault_dir.name]})

        for file_info in tqdm(all_files_to_process, desc="Extracting Features from PU"):
            try:
                file_path, fault_type = file_info['path'], file_info['type']
                match_rpm = re.search(r'N(\d{2})', file_path.stem)
                if not match_rpm: continue
                rpm = int(match_rpm.group(1)) * 100
                fault_freqs = self._calculate_fault_frequencies(rpm)
                data = scipy.io.loadmat(file_path)
                struct_key = [k for k in data.keys() if not k.startswith('__')][0]
                signal_data = data[struct_key]['Y'][0, 0][0, 6]['Data'].flatten()
                
                segment_length, step = 4096, 2048
                for i in range(0, len(signal_data) - segment_length, step):
                    segment = signal_data[i:i+segment_length]
                    features = self._extract_all_features_as_dict(segment, **fault_freqs)
                    sample = {'Fault_Type': fault_type, 'Fault_Severity': 'N/A', 
                              'source_file': file_path.name, **features}
                    all_samples_info.append(sample)
            except Exception as e:
                logging.error(f"An unexpected error occurred processing {file_path.name}. Error: {e}")
        
        if not all_samples_info:
            logging.warning("No data was loaded from PU."); return None
        return pd.DataFrame(all_samples_info)

    def _get_cwru_file_list(self):
        all_files = []; fault_dirs = [d for d in self.data_path.glob("*") if d.is_dir()]
        for fault_dir in fault_dirs:
            dirs_to_process = [{'path': fault_dir, 'severity_key': ''}] if fault_dir.name == 'Normal' else [{'path': sd, 'severity_key': sd.name} for sd in fault_dir.glob("*") if sd.is_dir()]
            for dir_info in dirs_to_process:
                for file_path in dir_info['path'].glob("**/*.mat"):
                    all_files.append({'path': file_path, 'fault_type_key': fault_dir.name, 'severity_key': dir_info['severity_key']})
        return all_files

    def _get_feature_names(self, wavelet_level=3):
        """
        Returns a list of ONLY the 20 core signal-processing-based feature names.
        This includes wavelet stats, envelope amplitudes, and permutation entropy.
        Prefixes and operational condition features are handled by the calling functions.
        """
        wavelet_stat_names = ['mean', 'std', 'kurtosis', 'skew']
        # Generates 4*4 = 16 wavelet features
        wavelet_names = [f'wavelet_{i}_{stat}' for i in range(wavelet_level + 1) for stat in wavelet_stat_names]
        
        # Generates 3 envelope features
        envelope_names = ['env_bpfi_amp', 'env_bpfo_amp', 'env_bsf_amp']
        
        # Adds the entropy feature name
        entropy_names = ['perm_entropy']
        
        return wavelet_names + envelope_names + entropy_names

    def _extract_all_features_as_dict(self, signal_segment, bpfi, bpfo, bsf, prefix=''):
        """
        Extracts all signal-derived features (20 total) and returns them as a dictionary.
        A prefix is added to all keys ONLY if a prefix is explicitly provided.
        """
        # --- 1. Extract base signal features using helper methods ---
        wavelet_features = self._extract_wavelet_stats(signal_segment)
        envelope_features = self._extract_envelope_features(signal_segment, bpfi, bpfo, bsf)
        
        # --- 2. Calculate entropy feature ---
        perm_entropy_feature = [0.0] # Default value in case of error
        if ent is not None:
            try:
                # order=3, delay=1 are common parameters.
                perm_entropy_feature = [ent.permutation_entropy(signal_segment, order=3, delay=1, normalize=True)]
            except Exception:
                # This can happen on very short or constant signals
                perm_entropy_feature = [0.0]
        
        # --- 3. Combine all signal-derived features ---
        combined_features = np.concatenate([
            wavelet_features, 
            envelope_features,
            perm_entropy_feature
        ])
        
        # --- 4. Get the corresponding feature names ---
        # This call now correctly gets the 20 base feature names
        feature_names = self._get_feature_names()

        # Final check to ensure lengths match, preventing zip errors
        if len(feature_names) != len(combined_features):
            logging.error(f"Feature name count ({len(feature_names)}) does not match value count ({len(combined_features)}).")
            return {}

        features = dict(zip(feature_names, combined_features))
            
        # --- 5. Apply prefix to all keys ONLY if a prefix is provided ---
        if prefix:
            return {f"{prefix}{key}": val for key, val in features.items()}
        else:
            return features
        
    def _calculate_fault_frequencies(self, rpm):
        if rpm == 0: return {'bpfi': 0, 'bpfo': 0, 'bsf': 0}
        shaft_speed_hz = rpm / 60
        cos_a = np.cos(self.contact_angle)
        d_p = self.ball_dia / self.pitch_dia
        bpfo = 0.5 * self.n_balls * shaft_speed_hz * (1 - d_p * cos_a)
        bpfi = 0.5 * self.n_balls * shaft_speed_hz * (1 + d_p * cos_a)
        bsf = 0.5 * (self.pitch_dia / self.ball_dia) * shaft_speed_hz * (1 - (d_p * cos_a)**2)
        return {'bpfi': bpfi, 'bpfo': bpfo, 'bsf': bsf}

    def _extract_envelope_features(self, signal_segment, bpfi, bpfo, bsf):
        if len(signal_segment) == 0: return np.array([0.0, 0.0, 0.0])
        analytic_signal = hilbert(signal_segment); envelope = np.abs(analytic_signal); N = len(envelope)
        yf = 2.0/N * np.abs(fft(envelope)[:N//2]); xf = fftfreq(N, 1 / self.fs)[:N//2]
        return np.array([self._find_amplitude_at_freq(xf, yf, bpfi), self._find_amplitude_at_freq(xf, yf, bpfo), self._find_amplitude_at_freq(xf, yf, bsf)])

    def _extract_wavelet_stats(self, signal, wavelet='db4', level=3):
        coeffs = pywt.wavedec(signal, wavelet, level=level); return np.array([stat for c in coeffs for stat in [np.mean(c), np.std(c), scipy.stats.kurtosis(c), scipy.stats.skew(c)]])

    def _find_amplitude_at_freq(self, freqs, amps, target_freq, tolerance=5):
        if target_freq == 0: return 0.0
        freq_indices = np.where((freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance))
        return np.max(amps[freq_indices]) if freq_indices[0].size > 0 else 0.0

    def get_text_column(self):
        """Returns the name of the column to be used as text input for the model."""
        return self.text_column

# === Independent Helper Functions ===
# These functions are moved out of the DataLoader class for better modularity.

def split_data(df, label_col, test_size=0.2, val_size=0.1, random_state=42):
    """Splits a DataFrame into training, validation, and test sets."""
    df_filtered = df[df[label_col] != -1].copy()
    if df_filtered.empty:
        logging.error(f"No valid data for label '{label_col}' after filtering. Cannot split.")
        return None, None, None
        
    stratify_by = df_filtered[label_col] if df_filtered[label_col].nunique() > 1 else None
    train_val_df, test_df = train_test_split(df_filtered, test_size=test_size, random_state=random_state, stratify=stratify_by)
    
    if len(train_val_df) < 2: # Need at least 2 samples to split again
        logging.error("Train-validation set is too small to split further. Adjust test_size or data.")
        return train_val_df, None, test_df # Return what we have, val_df will be None
        
    val_ratio = val_size / (1 - test_size)
    stratify_by_val = train_val_df[label_col] if train_val_df[label_col].nunique() > 1 else None
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=random_state, stratify=stratify_by_val)
    
    logging.info(f"Data split complete. Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, output_dir):
    """Saves the processed data splits to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_df is not None: train_df.to_csv(output_dir / "train.csv", index=False)
    if val_df is not None: val_df.to_csv(output_dir / "val.csv", index=False)
    if test_df is not None: test_df.to_csv(output_dir / "test.csv", index=False)
    logging.info(f"Saved processed data splits to {output_dir}")
