"""
Metrics calculation utilities for LLM Ripper.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates various metrics for model evaluation."""
    
    @staticmethod
    def compute_perplexity(
        model: torch.nn.Module,
        tokenizer: Any,
        texts: List[str],
        device: str = "cpu"
    ) -> float:
        """Compute perplexity on a list of texts."""
        
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                try:
                    inputs = tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512
                    ).to(device)
                    
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    num_tokens = inputs["input_ids"].shape[1]
                    
                    total_loss += loss * num_tokens
                    total_tokens += num_tokens
                    
                except Exception as e:
                    logger.warning(f"Error computing perplexity for text: {e}")
                    continue
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            return float(np.exp(avg_loss))
        else:
            return float('inf')
    
    @staticmethod
    def compute_embedding_similarity(
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between two embedding matrices."""
        
        if metric == "cosine":
            # Compute average cosine similarity
            similarities = []
            for i in range(min(len(embeddings1), len(embeddings2), 1000)):  # Sample for efficiency
                sim = torch.cosine_similarity(
                    embeddings1[i:i+1], 
                    embeddings2[i:i+1], 
                    dim=1
                ).item()
                similarities.append(sim)
            return float(np.mean(similarities))
        
        elif metric == "euclidean":
            # Compute average Euclidean distance (inverted for similarity)
            distances = []
            for i in range(min(len(embeddings1), len(embeddings2), 1000)):
                dist = torch.norm(embeddings1[i] - embeddings2[i]).item()
                distances.append(dist)
            avg_distance = np.mean(distances)
            return float(1.0 / (1.0 + avg_distance))  # Convert to similarity
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    @staticmethod
    def compute_attention_alignment(
        attention_maps1: torch.Tensor,
        attention_maps2: torch.Tensor
    ) -> float:
        """Compute alignment between attention maps."""
        
        # Flatten attention maps and compute correlation
        flat_attn1 = attention_maps1.flatten().cpu().numpy()
        flat_attn2 = attention_maps2.flatten().cpu().numpy()
        
        correlation, _ = pearsonr(flat_attn1, flat_attn2)
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def compute_cluster_metrics(
        activations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        n_clusters: int = 10
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(activations)
        
        metrics = {}
        
        # Silhouette score
        if len(np.unique(cluster_labels)) > 1:
            metrics["silhouette_score"] = float(silhouette_score(activations, cluster_labels))
        else:
            metrics["silhouette_score"] = 0.0
        
        # Inertia (within-cluster sum of squares)
        metrics["inertia"] = float(kmeans.inertia_)
        
        # If ground truth labels are provided
        if labels is not None:
            metrics["adjusted_rand_score"] = float(adjusted_rand_score(labels, cluster_labels))
            metrics["purity"] = MetricsCalculator._compute_purity(labels, cluster_labels)
        
        return metrics
    
    @staticmethod
    def _compute_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Compute cluster purity score."""
        
        total_samples = len(true_labels)
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        
        correctly_assigned = 0
        
        for cluster in unique_clusters:
            # Get true labels for this cluster
            cluster_mask = cluster_labels == cluster
            cluster_true_labels = true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                # Find most common true label in this cluster
                unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
                most_common_count = np.max(counts)
                correctly_assigned += most_common_count
        
        return correctly_assigned / total_samples if total_samples > 0 else 0.0
    
    @staticmethod
    def compute_classification_metrics(
        y_true: List[int],
        y_pred: List[int],
        average: str = "weighted"
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, average=average)),
        }
        
        # Matthews correlation coefficient for binary classification
        if len(set(y_true)) == 2:
            metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
        
        return metrics
    
    @staticmethod
    def compute_regression_metrics(
        y_true: List[float],
        y_pred: List[float]
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Mean squared error
        mse = float(np.mean((y_true - y_pred) ** 2))
        
        # Root mean squared error
        rmse = float(np.sqrt(mse))
        
        # Mean absolute error
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        # Pearson correlation
        pearson_r, _ = pearsonr(y_true, y_pred)
        
        # Spearman correlation
        spearman_r, _ = spearmanr(y_true, y_pred)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else 0.0,
            "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else 0.0
        }
    
    @staticmethod
    def compute_semantic_coverage(
        embeddings: torch.Tensor,
        vocab_size: int,
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """Compute semantic coverage metrics for embeddings."""
        
        # Sample embeddings for efficiency
        if len(embeddings) > sample_size:
            indices = torch.randperm(len(embeddings))[:sample_size]
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(sample_embeddings)):
            for j in range(i + 1, min(i + 50, len(sample_embeddings))):  # Limit pairs for efficiency
                sim = torch.cosine_similarity(
                    sample_embeddings[i:i+1], 
                    sample_embeddings[j:j+1]
                ).item()
                similarities.append(sim)
        
        similarities = np.array(similarities)
        
        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "semantic_diversity": float(1.0 - np.mean(similarities))  # Higher diversity = lower mean similarity
        }
    
    @staticmethod
    def compute_head_specialization_score(
        attention_weights: torch.Tensor,
        dependency_matrix: Optional[torch.Tensor] = None
    ) -> float:
        """Compute attention head specialization score."""
        
        # If dependency matrix is provided, compute alignment
        if dependency_matrix is not None:
            # Flatten and compute correlation
            attn_flat = attention_weights.flatten().cpu().numpy()
            dep_flat = dependency_matrix.flatten().cpu().numpy()
            
            correlation, _ = pearsonr(attn_flat, dep_flat)
            return float(correlation) if not np.isnan(correlation) else 0.0
        
        # Otherwise, compute attention pattern regularity
        # Higher entropy = less specialized
        attn_probs = torch.softmax(attention_weights, dim=-1)
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1)
        avg_entropy = torch.mean(entropy).item()
        
        # Convert to specialization score (lower entropy = higher specialization)
        max_entropy = np.log(attention_weights.shape[-1])
        specialization = 1.0 - (avg_entropy / max_entropy)
        
        return float(specialization)
    
    @staticmethod
    def compute_transplant_success_score(
        baseline_metrics: Dict[str, float],
        transplant_metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute overall transplant success score."""
        
        if weights is None:
            weights = {"accuracy": 0.3, "f1_score": 0.3, "perplexity": -0.2, "similarity": 0.2}
        
        success_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in baseline_metrics and metric in transplant_metrics:
                baseline_val = baseline_metrics[metric]
                transplant_val = transplant_metrics[metric]
                
                if weight > 0:  # Higher is better
                    improvement = (transplant_val - baseline_val) / baseline_val if baseline_val != 0 else 0
                else:  # Lower is better (e.g., perplexity)
                    improvement = (baseline_val - transplant_val) / baseline_val if baseline_val != 0 else 0
                
                success_score += abs(weight) * improvement
                total_weight += abs(weight)
        
        return success_score / total_weight if total_weight > 0 else 0.0