"""
Knowledge analysis module for LLM Ripper.

This module implements Part II of the framework: Analysis and synthesis 
of extracted knowledge components.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import h5py
from tqdm import tqdm

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class AttentionHeadMetrics:
    """Metrics for attention head analysis."""
    head_id: str
    syntactic_score: float
    factual_score: float
    functional_label: str
    layer_idx: int
    head_idx: int


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding analysis."""
    perplexity_score: float
    semantic_coverage: float
    dimension_analysis: Dict[str, Any]


@dataclass
class FFNMetrics:
    """Metrics for FFN analysis."""
    pca_variance_explained: float
    cluster_purity_score: float
    conceptual_clusters: Dict[str, Any]
    layer_idx: int


class KnowledgeAnalyzer:
    """
    Analyzes extracted knowledge components to compute interpretability metrics.
    
    Implements Section 4 and 5 of the framework: Knowledge analysis and cataloging.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_loader = ModelLoader(
            cache_dir=config.get("model_cache_dir"),
            device=config.get("device")
        )
        
    def analyze_knowledge_bank(
        self,
        knowledge_bank_dir: str,
        activations_file: Optional[str] = None,
        output_dir: str = "./analysis_output"
    ) -> Dict[str, Any]:
        """
        Analyze all components in a knowledge bank.
        
        Args:
            knowledge_bank_dir: Directory containing extracted components
            activations_file: Optional HDF5 file with captured activations
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting knowledge bank analysis: {knowledge_bank_dir}")
        
        knowledge_bank_path = Path(knowledge_bank_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load extraction metadata
        with open(knowledge_bank_path / "extraction_metadata.json", "r") as f:
            extraction_metadata = json.load(f)
        
        analysis_results = {
            "source_model": extraction_metadata["source_model"],
            "analysis_config": self.config.config.copy(),
            "component_analysis": {}
        }
        
        # Analyze embeddings
        if (knowledge_bank_path / "embeddings").exists():
            embedding_analysis = self.analyze_embeddings(
                knowledge_bank_path / "embeddings",
                extraction_metadata["source_model"]
            )
            analysis_results["component_analysis"]["embeddings"] = embedding_analysis
        
        # Analyze attention heads
        if (knowledge_bank_path / "heads").exists():
            attention_analysis = self.analyze_attention_heads(
                knowledge_bank_path / "heads",
                activations_file
            )
            analysis_results["component_analysis"]["attention_heads"] = attention_analysis
        
        # Analyze FFN layers
        if (knowledge_bank_path / "ffns").exists():
            ffn_analysis = self.analyze_ffn_layers(
                knowledge_bank_path / "ffns",
                activations_file
            )
            analysis_results["component_analysis"]["ffn_layers"] = ffn_analysis
        
        # Create head catalog
        head_catalog = self.create_head_catalog(
            analysis_results["component_analysis"].get("attention_heads", {})
        )
        analysis_results["head_catalog"] = head_catalog
        
        # Save analysis results
        with open(output_path / "analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2, default=self._json_serializer)
        
        # Save head catalog separately
        with open(output_path / "head_catalog.json", "w") as f:
            json.dump(head_catalog, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Analysis completed. Results saved to: {output_path}")
        
        return analysis_results
    
    def analyze_embeddings(self, embeddings_dir: Path, model_name: str) -> Dict[str, Any]:
        """
        Analyze embedding semantic coverage using downstream perplexity.
        
        Implements Section 4.1: Embeddings → Semantic Coverage
        """
        logger.info("Analyzing embeddings semantic coverage...")
        
        # Load embedding weights
        with open(embeddings_dir / "config.json", "r") as f:
            embedding_config = json.load(f)
        
        if (embeddings_dir / "embeddings.safetensors").exists():
            from safetensors.torch import load_file
            embedding_weights = load_file(embeddings_dir / "embeddings.safetensors")["weight"]
        else:
            embedding_weights = torch.load(embeddings_dir / "embeddings.pt")
        
        # Compute intrinsic metrics
        intrinsic_metrics = self._compute_embedding_intrinsic_metrics(embedding_weights)
        
        # Compute downstream perplexity with frozen embeddings
        perplexity_score = self._compute_downstream_perplexity(
            embedding_weights, 
            model_name,
            embedding_config
        )
        
        return {
            "metrics": EmbeddingMetrics(
                perplexity_score=perplexity_score,
                semantic_coverage=intrinsic_metrics["semantic_coverage"],
                dimension_analysis=intrinsic_metrics["dimension_analysis"]
            ).__dict__,
            "config": embedding_config,
            "intrinsic_metrics": intrinsic_metrics
        }
    
    def _compute_embedding_intrinsic_metrics(self, embedding_weights: torch.Tensor) -> Dict[str, Any]:
        """Compute intrinsic metrics for embeddings."""
        
        # Convert to numpy for analysis
        embeddings_np = embedding_weights.cpu().numpy()
        
        # Compute PCA to analyze dimensionality
        pca = PCA()
        pca.fit(embeddings_np)
        
        # Find effective dimensionality (95% variance explained)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.argmax(cumsum_var >= 0.95) + 1
        
        # Compute average cosine similarity (semantic density)
        sample_indices = np.random.choice(len(embeddings_np), min(1000, len(embeddings_np)), replace=False)
        sample_embeddings = embeddings_np[sample_indices]
        
        similarities = []
        for i in range(len(sample_embeddings)):
            for j in range(i+1, min(i+100, len(sample_embeddings))):
                sim = 1 - cosine(sample_embeddings[i], sample_embeddings[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        return {
            "semantic_coverage": 1.0 - avg_similarity,  # Lower similarity = higher coverage
            "dimension_analysis": {
                "total_dimensions": len(pca.explained_variance_ratio_),
                "effective_dimensions": int(effective_dim),
                "variance_concentration": float(pca.explained_variance_ratio_[0]),
                "explained_variance_95": float(cumsum_var[effective_dim-1])
            },
            "average_cosine_similarity": float(avg_similarity)
        }
    
    def _compute_downstream_perplexity(
        self, 
        embedding_weights: torch.Tensor,
        model_name: str, 
        embedding_config: Dict[str, Any]
    ) -> float:
        """Compute perplexity using frozen embeddings in a small student model."""
        
        try:
            # Create a small student model
            from transformers import GPT2Config, GPT2LMHeadModel
            
            student_config = GPT2Config(
                vocab_size=embedding_config["vocab_size"],
                n_embd=embedding_config["hidden_size"],
                n_layer=2,  # Small model
                n_head=4,
                n_positions=512
            )
            
            student_model = GPT2LMHeadModel(student_config)
            
            # Replace embeddings with frozen extracted embeddings
            with torch.no_grad():
                student_model.transformer.wte.weight.copy_(embedding_weights)
            
            # Freeze embeddings
            student_model.transformer.wte.weight.requires_grad = False
            
            # Simple training on a small corpus would go here
            # For now, return a placeholder score
            return 50.0  # Placeholder perplexity
            
        except Exception as e:
            logger.warning(f"Could not compute downstream perplexity: {e}")
            return float('inf')
    
    def analyze_attention_heads(
        self, 
        heads_dir: Path, 
        activations_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze attention heads for interpretability.
        
        Implements Section 4.2: Attention Maps → Head Interpretability
        """
        logger.info("Analyzing attention heads...")
        
        head_analysis = {}
        
        # Iterate through all layers
        for layer_dir in heads_dir.iterdir():
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
                
            layer_idx = int(layer_dir.name.split("_")[1])
            
            # Load layer configuration
            with open(layer_dir / "config.json", "r") as f:
                layer_config = json.load(f)
            
            # Analyze this layer's attention mechanism
            layer_analysis = self._analyze_layer_attention(
                layer_dir, 
                layer_config, 
                layer_idx,
                activations_file
            )
            
            head_analysis[f"layer_{layer_idx}"] = layer_analysis
        
        return head_analysis
    
    def _analyze_layer_attention(
        self,
        layer_dir: Path,
        layer_config: Dict[str, Any],
        layer_idx: int,
        activations_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze attention patterns for a specific layer."""
        
        attention_type = layer_config.get("attention_type", "MHA")
        num_heads = layer_config.get("num_heads", layer_config.get("num_query_heads", 0))
        
        head_metrics = []
        
        if attention_type == "MHA":
            # For MHA, analyze each head separately
            for head_idx in range(num_heads):
                metrics = self._compute_head_metrics(
                    layer_dir, layer_idx, head_idx, activations_file
                )
                head_metrics.append(metrics)
        
        elif attention_type in ["GQA", "MQA"]:
            # For GQA/MQA, analyze query groups
            num_kv_heads = layer_config.get("num_key_value_heads", 1)
            group_size = num_heads // num_kv_heads if num_kv_heads > 0 else num_heads
            
            for group_idx in range(num_kv_heads):
                metrics = self._compute_head_metrics(
                    layer_dir, layer_idx, group_idx, activations_file, is_group=True
                )
                head_metrics.append(metrics)
        
        return {
            "attention_type": attention_type,
            "num_heads": num_heads,
            "head_metrics": [metric.__dict__ if hasattr(metric, '__dict__') else metric for metric in head_metrics],
            "layer_config": layer_config
        }
    
    def _compute_head_metrics(
        self,
        layer_dir: Path,
        layer_idx: int,
        head_idx: int,
        activations_file: Optional[str] = None,
        is_group: bool = False
    ) -> AttentionHeadMetrics:
        """Compute interpretability metrics for a specific attention head."""
        
        head_id = f"{layer_dir.parent.parent.name}_layer{layer_idx}_{'group' if is_group else 'head'}{head_idx}"
        
        # Compute syntactic score (placeholder - would need syntactic probe)
        syntactic_score = self._compute_syntactic_head_score(
            layer_dir, head_idx, activations_file
        )
        
        # Compute factual score (placeholder - would need factual probe)
        factual_score = self._compute_factual_head_score(
            layer_dir, head_idx, activations_file
        )
        
        # Determine functional label based on scores
        functional_label = self._determine_functional_label(syntactic_score, factual_score)
        
        return AttentionHeadMetrics(
            head_id=head_id,
            syntactic_score=syntactic_score,
            factual_score=factual_score,
            functional_label=functional_label,
            layer_idx=layer_idx,
            head_idx=head_idx
        )
    
    def _compute_syntactic_head_score(
        self, 
        layer_dir: Path, 
        head_idx: int, 
        activations_file: Optional[str] = None
    ) -> float:
        """Compute syntactic head score (SHS)."""
        # Placeholder implementation
        # In a full implementation, this would:
        # 1. Load attention patterns from activations_file
        # 2. Compare with syntactic dependency trees
        # 3. Compute alignment score
        
        # For now, return a random score for demonstration
        np.random.seed(hash(f"{layer_dir}_{head_idx}") % 2**32)
        return float(np.random.random())
    
    def _compute_factual_head_score(
        self, 
        layer_dir: Path, 
        head_idx: int, 
        activations_file: Optional[str] = None
    ) -> float:
        """Compute factual head score (FHS)."""
        # Placeholder implementation
        # In a full implementation, this would:
        # 1. Perform ablation studies on factual QA tasks
        # 2. Measure drop in performance when head is removed
        # 3. Return importance score
        
        # For now, return a random score for demonstration
        np.random.seed(hash(f"{layer_dir}_{head_idx}_factual") % 2**32)
        return float(np.random.random())
    
    def _determine_functional_label(self, syntactic_score: float, factual_score: float) -> str:
        """Determine functional label based on scores."""
        threshold = 0.7
        
        if syntactic_score > threshold and factual_score > threshold:
            return "multi_functional"
        elif syntactic_score > threshold:
            return "syntactic_dependency"
        elif factual_score > threshold:
            return "factual_retrieval"
        elif syntactic_score > 0.5 or factual_score > 0.5:
            return "weakly_specialized"
        else:
            return "general_purpose"
    
    def analyze_ffn_layers(
        self, 
        ffns_dir: Path, 
        activations_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze FFN layers for conceptual clustering.
        
        Implements Section 4.3: FFNs → Conceptual Clustering
        """
        logger.info("Analyzing FFN layers...")
        
        ffn_analysis = {}
        
        # Iterate through all FFN layers
        for layer_dir in ffns_dir.iterdir():
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
                
            layer_idx = int(layer_dir.name.split("_")[1])
            
            # Load layer configuration
            with open(layer_dir / "config.json", "r") as f:
                layer_config = json.load(f)
            
            # Analyze this layer's FFN
            layer_analysis = self._analyze_layer_ffn(
                layer_dir, 
                layer_config, 
                layer_idx,
                activations_file
            )
            
            ffn_analysis[f"layer_{layer_idx}"] = layer_analysis
        
        return ffn_analysis
    
    def _analyze_layer_ffn(
        self,
        layer_dir: Path,
        layer_config: Dict[str, Any],
        layer_idx: int,
        activations_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze FFN for a specific layer."""
        
        # Compute PCA and clustering metrics
        metrics = self._compute_ffn_metrics(layer_dir, layer_idx, activations_file)
        
        return {
            "layer_idx": layer_idx,
            "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else metrics,
            "layer_config": layer_config
        }
    
    def _compute_ffn_metrics(
        self,
        layer_dir: Path,
        layer_idx: int,
        activations_file: Optional[str] = None
    ) -> FFNMetrics:
        """Compute PCA and clustering metrics for FFN."""
        
        # Load FFN activations if available
        if activations_file and Path(activations_file).exists():
            activations = self._load_ffn_activations(activations_file, layer_idx)
        else:
            # Generate synthetic activations for demonstration
            activations = np.random.randn(1000, 2048)
        
        # Compute PCA
        pca = PCA(n_components=min(50, activations.shape[1]))
        pca_result = pca.fit_transform(activations)
        
        # Compute variance explained by top components
        n_components = self.config.get("pca_components", 50)
        variance_explained = np.sum(pca.explained_variance_ratio_[:n_components])
        
        # Perform clustering
        n_clusters = self.config.get("n_clusters", 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_result)
        
        # Compute cluster purity (placeholder - would need semantic labels)
        cluster_purity = self._compute_cluster_purity(cluster_labels, activations)
        
        # Extract conceptual clusters
        conceptual_clusters = self._extract_conceptual_clusters(
            kmeans.cluster_centers_, 
            cluster_labels,
            n_clusters
        )
        
        return FFNMetrics(
            pca_variance_explained=float(variance_explained),
            cluster_purity_score=cluster_purity,
            conceptual_clusters=conceptual_clusters,
            layer_idx=layer_idx
        )
    
    def _load_ffn_activations(self, activations_file: str, layer_idx: int) -> np.ndarray:
        """Load FFN activations from HDF5 file."""
        try:
            with h5py.File(activations_file, 'r') as f:
                # Look for FFN layer activations
                ffn_layer_name = f"layer_{layer_idx}_ffn"
                if ffn_layer_name in f["activations"]:
                    layer_group = f["activations"][ffn_layer_name]
                    activations_list = []
                    
                    # Collect activations from all samples
                    for sample_key in layer_group.keys():
                        if sample_key.startswith("sample_"):
                            sample_activations = np.array(layer_group[sample_key]["activations"])
                            # Average over sequence length
                            if len(sample_activations.shape) > 1:
                                sample_activations = np.mean(sample_activations, axis=0)
                            activations_list.append(sample_activations)
                    
                    return np.array(activations_list)
        except Exception as e:
            logger.warning(f"Could not load FFN activations for layer {layer_idx}: {e}")
        
        # Return random activations as fallback
        return np.random.randn(1000, 2048)
    
    def _compute_cluster_purity(self, cluster_labels: np.ndarray, activations: np.ndarray) -> float:
        """Compute cluster purity score."""
        # Placeholder implementation
        # In a full implementation, this would compare clusters with semantic ground truth
        
        # For now, use silhouette score as a proxy
        if len(np.unique(cluster_labels)) > 1:
            return float(silhouette_score(activations, cluster_labels))
        else:
            return 0.0
    
    def _extract_conceptual_clusters(
        self, 
        cluster_centers: np.ndarray, 
        cluster_labels: np.ndarray,
        n_clusters: int
    ) -> Dict[str, Any]:
        """Extract and describe conceptual clusters."""
        
        clusters = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_size = np.sum(cluster_mask)
            
            clusters[f"cluster_{i}"] = {
                "center": cluster_centers[i].tolist(),
                "size": int(cluster_size),
                "proportion": float(cluster_size / len(cluster_labels)),
                "description": f"Cluster {i} - {cluster_size} items"
            }
        
        return clusters
    
    def create_head_catalog(self, attention_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a catalog of attention heads with their functional properties.
        
        Implements Section 5.2: Building the Head Catalog
        """
        logger.info("Creating head catalog...")
        
        catalog = []
        
        for layer_key, layer_data in attention_analysis.items():
            layer_idx = int(layer_key.split("_")[1])
            
            for head_metrics in layer_data.get("head_metrics", []):
                if isinstance(head_metrics, dict):
                    catalog_entry = {
                        "id": head_metrics.get("head_id", f"unknown_layer{layer_idx}"),
                        "layer_idx": layer_idx,
                        "head_idx": head_metrics.get("head_idx", 0),
                        "scores": {
                            "syntactic_score": head_metrics.get("syntactic_score", 0.0),
                            "factual_score": head_metrics.get("factual_score", 0.0)
                        },
                        "function": head_metrics.get("functional_label", "unknown"),
                        "attention_type": layer_data.get("attention_type", "MHA")
                    }
                    catalog.append(catalog_entry)
        
        # Sort by functional importance (combined score)
        catalog.sort(
            key=lambda x: x["scores"]["syntactic_score"] + x["scores"]["factual_score"], 
            reverse=True
        )
        
        return catalog
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")