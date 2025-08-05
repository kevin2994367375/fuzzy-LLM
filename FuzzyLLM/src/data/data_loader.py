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

class DataLoader:
    """
    Its ONLY responsibility is to load raw files and extract features into a DataFrame
    with human-readable string labels (e.g., 'Inner Race Fault').
    It does NOT handle any numeric encoding of labels.
    """
    def __init__(self, data_dir, dataset_name='cwru'):
        self.raw_data_dir = Path(data_dir) if data_dir else None
        self.data_path = self.raw_data_dir / dataset_name.upper() if self.raw_data_dir else None
        self.dataset_name = dataset_name.lower()
        self._load_dataset_parameters()
        if self.data_path:
            logging.info(f"DataLoader configured for {self.dataset_name.upper()}.")
        
    def _load_dataset_parameters(self):
        """Loads physical parameters for the supported datasets."""
        if self.dataset_name == 'cwru':
            self.fs = 12000
            self.rpm_map = {'0': 1797, '1': 1772, '2': 1750, '3': 1730}
            self.default_rpm = 1797
            self.n_balls, self.ball_dia, self.pitch_dia, self.contact_angle = 9, 0.3126 * 25.4e-3, 1.516 * 25.4e-3, np.deg2rad(0)
        elif self.dataset_name == 'hust':
            self.fs = 25600
            self.n_balls, self.ball_dia, self.pitch_dia, self.contact_angle = 9, 7.94e-3, 1.516 * 25.4e-3, np.deg2rad(0)
        elif self.dataset_name == 'pu':
            logging.info("Loading VERIFIED parameters for PU dataset (6203 Bearing).")
            self.fs = 64000
            self.n_balls, self.ball_dia, self.pitch_dia, self.contact_angle = 8, 9.525e-3, 28.5e-3, np.deg2rad(0)

    def load_and_extract_features(self):
        """Public dispatcher method to load data and extract features."""
        if self.data_path is None or not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path is not valid: {self.data_path}")
        logging.info(f"Loading data and extracting features from: {self.data_path}")
        if self.dataset_name == 'cwru':
            return self._process_cwru_data()
        elif self.dataset_name == 'pu':
            return self._process_pu_data()
        elif self.dataset_name == 'hust':
            return self._process_hust_data()
        logging.warning(f"No data processing method found for dataset: {self.dataset_name}")
        return None

    def _process_hust_data(self):
        """Processes HUST data, fusing features from all three (X, Y, Z) vibration channels."""
        all_samples_info = []
        location_map = {'C': 'Normal', 'I': 'Inner Race Fault', 'O': 'Outer Race Fault', 'B': 'Ball Fault'}
        severity_map = {'0.5X': 'Slight', 'X': 'Severe'}
        
        xls_files = list(self.data_path.rglob("*.xls*"))
        if not xls_files:
            logging.error(f"No .xls or .xlsx files found in {self.data_path}."); return None

        for file_path in tqdm(xls_files, desc="Processing HUST Bearings"):
            try:
                filename = file_path.stem
                parts = filename.split('_')
                if len(parts) < 2: continue
                fault_location, fault_severity, op_speed_hz, op_load_val = None, 'N/A', 0.0, 0.0
                if len(parts) == 3 and parts[1] in location_map:
                    fault_severity = severity_map.get(parts[0], 'Severe')
                    fault_location = location_map.get(parts[1])
                    op_speed_hz = float(re.search(r'(\d+)', parts[2]).group(1))
                    op_load_val = float(re.search(r'(\d+\.?\d*)', parts[0]).group(1)) if 'X' in parts[0] else 0
                elif len(parts) == 2 and parts[0] in location_map:
                    fault_location = location_map.get(parts[0])
                    if fault_location != 'Normal': fault_severity = 'Severe'
                    op_speed_hz = float(re.search(r'(\d+)', parts[1]).group(1))
                else:
                    continue
                if fault_location is None: continue

                df_signals = pd.read_csv(file_path, sep=r'\s+', header=None, skiprows=22, on_bad_lines='skip')
                if df_signals.shape[1] < 5: continue
                signal_x = df_signals.iloc[:, 2].astype(np.float64).values
                signal_y = df_signals.iloc[:, 3].astype(np.float64).values
                signal_z = df_signals.iloc[:, 4].astype(np.float64).values

                segment_length, step = 4096, 2048
                min_len = min(len(signal_x), len(signal_y), len(signal_z))
                
                for j in range(0, min_len - segment_length, step):
                    features_x = self._extract_all_features_as_dict(signal_x[j:j+segment_length], 0, 0, 0, prefix='x_')
                    features_y = self._extract_all_features_as_dict(signal_y[j:j+segment_length], 0, 0, 0, prefix='y_')
                    features_z = self._extract_all_features_as_dict(signal_z[j:j+segment_length], 0, 0, 0, prefix='z_')
                    op_features = {'op_speed': op_speed_hz, 'op_load': op_load_val}
                    combined_features = {**features_x, **features_y, **features_z, **op_features}
                    
                    sample = {'Fault_Type': fault_location, 'Fault_Severity': fault_severity,
                              'source_file': file_path.name, 'bearing_id': filename, **combined_features}
                    all_samples_info.append(sample)
            except Exception as e:
                logging.error(f"Error processing HUST file {file_path}: {e}")

        return pd.DataFrame(all_samples_info) if all_samples_info else None

    def _process_cwru_data(self):
        """Processes CWRU data from .mat files."""
        all_samples_info = []
        fault_type_map = {'B': 'Ball Fault', 'IR': 'Inner Race Fault', 'OR': 'Outer Race Fault', 'Normal': 'Normal'}
        severity_map = {'007': 'Slight', '014': 'Moderate', '021': 'Severe', '028': 'Critical'}
        
        for file_info in tqdm(self._get_cwru_file_list(), desc="Extracting Features from CWRU"):
            try:
                file_path = file_info['path']
                match = re.search(r'_(\d)\.mat$', file_path.name)
                rpm = self.rpm_map.get(match.group(1) if match else '0', self.default_rpm)
                fault_freqs = self._calculate_fault_frequencies(rpm)
                mat_data = scipy.io.loadmat(file_path)
                signal_key = next((k for k in mat_data if 'DE_time' in k), None)
                if not signal_key: continue
                
                signal_data = mat_data[signal_key].flatten()
                segment_length, step = 4096, 2048
                for i in range(0, len(signal_data) - segment_length, step):
                    segment = signal_data[i:i+segment_length]
                    features = self._extract_all_features_as_dict(segment, **fault_freqs)
                    sample = {
                        'Fault_Type': fault_type_map[file_info['fault_type_key']],
                        'Fault_Severity': severity_map.get(file_info['severity_key'], 'N/A'),
                        'source_file': file_path.name, **features
                    }
                    all_samples_info.append(sample)
            except Exception as e:
                logging.error(f"Error processing CWRU file {file_path}: {e}")
        
        return pd.DataFrame(all_samples_info) if all_samples_info else None

    def _process_pu_data(self):
        """Processes Paderborn University (PU) data from .mat files."""
        all_samples_info = []
        folder_to_label_map = {"Healthy": "Normal", "Outer Ring Fault": "Outer Race Fault", 
                               "Inner Ring Fault": "Inner Race Fault", "Compound Fault": "Compound Fault"}
        
        files_to_process = [{'path': file_path, 'type': folder_to_label_map[fault_dir.name]}
                            for fault_dir in self.data_path.glob("*") if fault_dir.is_dir() and fault_dir.name in folder_to_label_map
                            for file_path in fault_dir.glob("**/*.mat")]

        for file_info in tqdm(files_to_process, desc="Extracting Features from PU"):
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
                logging.error(f"Error processing PU file {file_path.name}: {e}")
        
        return pd.DataFrame(all_samples_info) if all_samples_info else None

    def _get_cwru_file_list(self):
        """Helper to generate a list of all CWRU .mat files to process."""
        all_files = []
        for fault_dir in self.data_path.glob("*"):
            if not fault_dir.is_dir(): continue
            dirs_to_process = [{'path': fault_dir, 'severity_key': ''}] if fault_dir.name == 'Normal' else \
                              [{'path': sd, 'severity_key': sd.name} for sd in fault_dir.glob("*") if sd.is_dir()]
            for dir_info in dirs_to_process:
                for file_path in dir_info['path'].glob("**/*.mat"):
                    all_files.append({'path': file_path, 'fault_type_key': fault_dir.name, 'severity_key': dir_info['severity_key']})
        return all_files

    def _get_feature_names(self, wavelet_level=3):
        """Returns a list of the 20 core signal-processing-based feature names."""
        wavelet_stats = ['mean', 'std', 'kurtosis', 'skew']
        wavelet_names = [f'wavelet_{i}_{stat}' for i in range(wavelet_level + 1) for stat in wavelet_stats]
        envelope_names = ['env_bpfi_amp', 'env_bpfo_amp', 'env_bsf_amp']
        entropy_names = ['perm_entropy']
        return wavelet_names + envelope_names + entropy_names

    def _extract_all_features_as_dict(self, signal_segment, bpfi, bpfo, bsf, prefix=''):
        """Extracts all signal-derived features and returns them as a dictionary."""
        wavelet_features = self._extract_wavelet_stats(signal_segment)
        envelope_features = self._extract_envelope_features(signal_segment, bpfi, bpfo, bsf)
        
        try:
            perm_entropy_feature = [ent.permutation_entropy(signal_segment, order=3, delay=1, normalize=True)]
        except Exception:
            perm_entropy_feature = [0.0]
        
        combined_features = np.concatenate([wavelet_features, envelope_features, perm_entropy_feature])
        feature_names = self._get_feature_names()
        
        if len(feature_names) != len(combined_features):
            logging.error(f"Feature name/value count mismatch.")
            return {}
            
        features = dict(zip(feature_names, combined_features))
        return {f"{prefix}{key}": val for key, val in features.items()} if prefix else features
        
    def _calculate_fault_frequencies(self, rpm):
        """Calculates characteristic fault frequencies based on bearing geometry and speed."""
        if rpm == 0: return {'bpfi': 0, 'bpfo': 0, 'bsf': 0}
        shaft_speed_hz = rpm / 60
        cos_a = np.cos(self.contact_angle)
        d_p = self.ball_dia / self.pitch_dia
        bpfo = 0.5 * self.n_balls * shaft_speed_hz * (1 - d_p * cos_a)
        bpfi = 0.5 * self.n_balls * shaft_speed_hz * (1 + d_p * cos_a)
        bsf = 0.5 * (self.pitch_dia / self.ball_dia) * shaft_speed_hz * (1 - (d_p * cos_a)**2)
        return {'bpfi': bpfi, 'bpfo': bpfo, 'bsf': bsf}

    def _extract_envelope_features(self, signal_segment, bpfi, bpfo, bsf):
        """Extracts amplitudes from the envelope spectrum at characteristic frequencies."""
        if len(signal_segment) == 0: return np.zeros(3)
        analytic_signal = hilbert(signal_segment)
        envelope = np.abs(analytic_signal)
        N = len(envelope)
        yf = 2.0/N * np.abs(fft(envelope)[:N//2])
        xf = fftfreq(N, 1 / self.fs)[:N//2]
        return np.array([self._find_amplitude_at_freq(xf, yf, bpfi),
                         self._find_amplitude_at_freq(xf, yf, bpfo),
                         self._find_amplitude_at_freq(xf, yf, bsf)])

    def _extract_wavelet_stats(self, signal, wavelet='db4', level=3):
        """Extracts statistical features from wavelet decomposition coefficients."""
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        return np.array([stat for c in coeffs for stat in [np.mean(c), np.std(c), scipy.stats.kurtosis(c), scipy.stats.skew(c)]])

    def _find_amplitude_at_freq(self, freqs, amps, target_freq, tolerance=5):
        """Finds the maximum amplitude within a tolerance band around a target frequency."""
        if target_freq == 0: return 0.0
        freq_indices = np.where((freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance))
        return np.max(amps[freq_indices]) if freq_indices[0].size > 0 else 0.0

# === Independent Helper Functions ===
# These functions are used by main.py for data splitting and saving.

def split_data(df, label_col, test_size=0.2, val_size=0.1, random_state=42):
    """Splits a DataFrame into training, validation, and test sets."""
    df_filtered = df[df[label_col] != -1].copy()
    if df_filtered.empty:
        logging.error(f"No valid data for label '{label_col}' after filtering. Cannot split.")
        return None, None, None
        
    stratify_by = df_filtered[label_col] if df_filtered[label_col].nunique() > 1 else None
    train_val_df, test_df = train_test_split(df_filtered, test_size=test_size, random_state=random_state, stratify=stratify_by)
    
    if len(train_val_df) < 2:
        logging.warning("Train-validation set is too small to split further.")
        return train_val_df, None, test_df
        
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
