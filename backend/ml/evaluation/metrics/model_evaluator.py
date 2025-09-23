"""
Model Evaluation Metrics

モデル評価指標の実装
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)


class ModelEvaluator:
    """モデル評価クラス"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(
        self, 
        true_labels: List[int], 
        predictions: List[float],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """包括的な評価指標計算"""
        
        # 予測値を二値化
        binary_predictions = [1 if p >= threshold else 0 for p in predictions]
        
        # 基本指標
        metrics = {
            'auc_roc': roc_auc_score(true_labels, predictions),
            'auc_pr': average_precision_score(true_labels, predictions),
            'accuracy': accuracy_score(true_labels, binary_predictions),
            'precision': precision_score(true_labels, binary_predictions, zero_division=0),
            'recall': recall_score(true_labels, binary_predictions, zero_division=0),
            'f1_score': f1_score(true_labels, binary_predictions, zero_division=0)
        }
        
        # 混同行列
        cm = confusion_matrix(true_labels, binary_predictions)
        metrics['confusion_matrix'] = cm
        
        # 分類レポート
        report = classification_report(
            true_labels, 
            binary_predictions, 
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # 閾値別性能（複数閾値での評価）
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_metrics = {}
        
        for t in thresholds:
            t_binary_pred = [1 if p >= t else 0 for p in predictions]
            threshold_metrics[f'threshold_{t}'] = {
                'accuracy': accuracy_score(true_labels, t_binary_pred),
                'precision': precision_score(true_labels, t_binary_pred, zero_division=0),
                'recall': recall_score(true_labels, t_binary_pred, zero_division=0),
                'f1_score': f1_score(true_labels, t_binary_pred, zero_division=0)
            }
        
        metrics['threshold_analysis'] = threshold_metrics
        
        return metrics
    
    def calculate_ranking_metrics(
        self, 
        true_labels: List[int], 
        predictions: List[float],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """ランキング評価指標"""
        
        # ソート（予測値降順）
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_true_labels = [true_labels[i] for i in sorted_indices]
        
        ranking_metrics = {}
        
        for k in k_values:
            if k > len(sorted_true_labels):
                continue
                
            # Precision@K
            top_k_labels = sorted_true_labels[:k]
            precision_at_k = sum(top_k_labels) / k
            ranking_metrics[f'precision_at_{k}'] = precision_at_k
            
            # Recall@K
            total_positive = sum(true_labels)
            if total_positive > 0:
                recall_at_k = sum(top_k_labels) / total_positive
                ranking_metrics[f'recall_at_{k}'] = recall_at_k
            else:
                ranking_metrics[f'recall_at_{k}'] = 0.0
        
        # NDCG計算
        for k in k_values:
            if k > len(sorted_true_labels):
                continue
            ndcg_k = self._calculate_ndcg(sorted_true_labels[:k])
            ranking_metrics[f'ndcg_at_{k}'] = ndcg_k
        
        return ranking_metrics
    
    def _calculate_ndcg(self, relevance_scores: List[int]) -> float:
        """NDCG (Normalized Discounted Cumulative Gain) 計算"""
        if not relevance_scores:
            return 0.0
        
        # DCG計算
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        # IDCG計算（理想的なDCG）
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = ideal_relevance[0]
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1)
        
        # NDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def calculate_business_metrics(
        self, 
        true_labels: List[int], 
        predictions: List[float],
        conversion_value: float = 1.0
    ) -> Dict[str, float]:
        """ビジネス指標計算"""
        
        # 予測値でソート（降順）
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_true_labels = [true_labels[i] for i in sorted_indices]
        sorted_predictions = [predictions[i] for i in sorted_indices]
        
        total_items = len(true_labels)
        total_positive = sum(true_labels)
        
        # カバレッジ分析（推薦範囲別の性能）
        coverage_metrics = {}
        coverage_ranges = [0.1, 0.2, 0.3, 0.5]
        
        for coverage in coverage_ranges:
            n_items = int(total_items * coverage)
            if n_items == 0:
                continue
                
            covered_labels = sorted_true_labels[:n_items]
            covered_positive = sum(covered_labels)
            
            coverage_metrics[f'coverage_{int(coverage*100)}pct'] = {
                'items_covered': n_items,
                'positive_captured': covered_positive,
                'capture_rate': covered_positive / total_positive if total_positive > 0 else 0,
                'precision': covered_positive / n_items if n_items > 0 else 0
            }
        
        # 期待価値計算
        expected_value = sum(p * conversion_value for p in predictions)
        actual_value = sum(true_labels) * conversion_value
        
        business_metrics = {
            'expected_value': expected_value,
            'actual_value': actual_value,
            'value_accuracy': min(expected_value / actual_value, actual_value / expected_value) if actual_value > 0 else 0,
            'coverage_analysis': coverage_metrics
        }
        
        return business_metrics