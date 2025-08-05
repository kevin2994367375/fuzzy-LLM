# src/data/fuzzy_encoder.py

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging
from sklearn.cluster import KMeans
from tqdm import tqdm

# Register tqdm for pandas apply
tqdm.pandas(desc="Processing Samples")

class AdaptiveFuzzyEncoder:
    """
    A self-adaptive fuzzy system that performs two major tasks:
    1. Dynamically generates membership functions based on data statistics.
    2. Encodes numerical data into human-readable fuzzy text.
    (Note: The prototype and soft-label generation methods are kept for potential future use)
    """
    def __init__(self, numeric_df, labels=None, config=None): # Made labels and config optional
        """
        :param numeric_df: DataFrame with ONLY the numerical features for analysis.
        :param labels: A pandas Series with the ground truth labels (numeric), used for grouping.
        :param config: The configuration for adaptive MF generation.
        """
        logging.info("Initializing Self-Adaptive Fuzzy System...")
        if not isinstance(numeric_df, pd.DataFrame):
            raise TypeError("Input 'numeric_df' must be a pandas DataFrame.")
            
        self.numeric_df = numeric_df
        self.labels = labels
        self.config = config
        self.fis_inputs = {}
        
        # Only create MFs if config is provided
        if self.config:
            for feature_name in self.numeric_df.columns:
                self._create_adaptive_mfs(feature_name)
            logging.info("Adaptive Membership Functions created successfully.")

    def _create_adaptive_mfs(self, feature_name):
        series = self.numeric_df[feature_name]
        q_min, q_max = series.min(), series.max()
        buffer = (q_max - q_min) * 0.1 if q_max > q_min else 1.0
        universe = np.linspace(q_min - buffer, q_max + buffer, 300)
        var = ctrl.Antecedent(universe, feature_name)
        
        for mf_conf in self.config['membership_functions']:
            mf_name, strategy, params = mf_conf['name'], mf_conf.get('strategy', 'quantile'), mf_conf['params']
            if strategy == 'quantile':
                points = series.quantile(params).values
                var[mf_name] = fuzz.trimf(var.universe, points)
            elif strategy == 'mean_std':
                mean, std = series.mean(), series.std()
                center = mean + params['center_offset'] * std
                width = params['width_factor'] * std
                var[mf_name] = fuzz.trimf(var.universe, [center - width, center, center + width])
        self.fis_inputs[feature_name] = var

    # 【【【 核心修改：升级文本生成逻辑 】】】
    def encode_row_to_text(self, row):
        """
        Encodes a single row to a fuzzy descriptive string,
        now embedding the raw numerical value.
        Example: 'kurtosis (8.9100) is Very High'
        """
        if not self.fis_inputs:
             raise RuntimeError("Fuzzy system not initialized with a config. Cannot encode.")

        fuzzy_descriptions = []
        for feature, value in row.items():
            if feature in self.fis_inputs:
                fis_input_var = self.fis_inputs[feature]
                memberships = {mf_name: fuzz.interp_membership(fis_input_var.universe, fis_input_var[mf_name].mf, value)
                               for mf_name in fis_input_var.terms}
                
                # 只选择隶属度最高的那个模糊集作为“专家判断”
                if memberships:
                    best_mf_name = max(memberships, key=memberships.get)
                    
                    # 简化特征名和模糊集名，使其更像自然语言
                    clean_feature_name = feature.split('_')[-1]
                    clean_mf_name = best_mf_name.replace('_', ' ')
                    
                    # 构建新的、信息融合的描述字符串
                    # 我们将原始数值格式化为保留4位小数的浮点数
                    description = f"{clean_feature_name} ({value:.4f}) is {clean_mf_name}"
                    fuzzy_descriptions.append(description)

        # 用分号和空格连接所有特征的描述
        return "; ".join(fuzzy_descriptions) + "."

    # 【【【 方法2：核心修改 - 新的 batch_encode_to_text 】】】
    # 它现在调用上面的方法，并增加了 return_series 功能
    def batch_encode_to_text(self, df, return_series=False):
        """
        Applies fuzzy text encoding to an entire DataFrame.

        Args:
            df (pd.DataFrame): The input dataframe with numerical features.
            return_series (bool): If True, returns a Pandas Series. Otherwise, returns a list.

        Returns:
            list or pd.Series: The fuzzy text representations.
        """
        logging.info(f"Generating fuzzy text representation for {len(df)} samples...")
        
        # 使用 progress_apply (因为您安装了tqdm.pandas)
        fuzzy_texts_series = df.progress_apply(self.encode_row_to_text, axis=1)
        
        if return_series:
            return fuzzy_texts_series
        else:
            return fuzzy_texts_series.tolist()

    # --- 以下是您原来的原型和软标签方法，我们保留它们，但当前的主流程不使用它们 ---
    
    def generate_prototypes(self, proto_config):
        """Generates class prototypes based on the specified strategy."""
        # ... (此方法的代码保持完全不变) ...
        strategy = proto_config['strategy']
        params = proto_config.get('params', {})
        logging.info(f"Generating class prototypes using '{strategy}' strategy...")

        unique_labels = sorted(self.labels.unique())
        for label in unique_labels:
            class_samples = self.numeric_df[self.labels == label]
            if class_samples.empty: continue

            if strategy == 'median':
                prototype = class_samples.median().to_frame().T
                self.class_prototypes[label] = [prototype]
            elif strategy == 'kmeans':
                k = params.get('k', 3)
                if len(class_samples) < k:
                    logging.warning(f"Class {label}: not enough samples ({len(class_samples)}) for {k} clusters. Using median instead.")
                    self.class_prototypes[label] = [class_samples.median().to_frame().T]
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(class_samples)
                self.class_prototypes[label] = [pd.DataFrame([center], columns=self.numeric_df.columns) for center in kmeans.cluster_centers_]

        logging.info("Pre-calculating fuzzy vectors for all prototypes...")
        self.proto_fuzzy_vectors = {label: [self._get_fuzzy_vector(proto.iloc[0]) for proto in protos]
                                    for label, protos in self.class_prototypes.items()}
    
    def _get_fuzzy_vector(self, row):
        # ... (此方法的代码保持完全不变) ...
        fuzzy_vector = {}
        for feature, value in row.items():
            if feature in self.fis_inputs:
                fis_input_var = self.fis_inputs[feature]
                memberships = np.array([fuzz.interp_membership(fis_input_var.universe, fis_input_var[mf_name].mf, value)
                                        for mf_name in fis_input_var.terms])
                fuzzy_vector[feature] = memberships / memberships.sum() if memberships.sum() > 0 else memberships
        return fuzzy_vector
        
    def _calculate_similarity(self, sample_fuzz_vec, proto_fuzz_vec):
        # ... (此方法的代码保持完全不变) ...
        total_similarity = 0.0
        common_features = sample_fuzz_vec.keys() & proto_fuzz_vec.keys()
        if not common_features: return 0.0
        
        for feature in common_features:
            total_similarity += np.dot(sample_fuzz_vec[feature], proto_fuzz_vec[feature])
        return total_similarity / len(common_features)
        
    def generate_soft_label(self, row):
        # ... (此方法的代码保持完全不变) ...
        if not self.proto_fuzzy_vectors:
            raise RuntimeError("Prototypes have not been generated. Call generate_prototypes() first.")
            
        sample_fuzz_vec = self._get_fuzzy_vector(row)
        
        class_scores = {}
        sorted_labels = sorted(self.proto_fuzzy_vectors.keys())
        
        for label in sorted_labels:
            proto_fuzz_vecs = self.proto_fuzzy_vectors.get(label, [])
            max_sim_for_class = 0
            if proto_fuzz_vecs:
                max_sim_for_class = max(self._calculate_similarity(sample_fuzz_vec, p_vec) for p_vec in proto_fuzz_vecs)
            class_scores[label] = max_sim_for_class
            
        scores_array = np.array(list(class_scores.values()))
        softmax_probs = np.exp(scores_array) / np.sum(np.exp(scores_array))
        
        return dict(zip(sorted_labels, softmax_probs))
        
    def batch_generate_soft_labels(self, df):
        """Applies soft label generation to an entire DataFrame."""
        logging.info(f"Generating fuzzy soft labels for {len(df)} samples...")
        return df.progress_apply(self.generate_soft_label, axis=1)