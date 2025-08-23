"""
Visualization utilities for LLM Ripper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization utilities for analysis results."""
    
    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (10, 8)):
        sns.set_style(style)
        self.figsize = figsize
        
    def plot_embedding_space(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = "umap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot 2D projection of embedding space."""
        
        # Reduce dimensionality
        if method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if labels:
            # Color by labels if provided
            unique_labels = list(set(labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax.scatter(
                    embedding_2d[mask, 0], 
                    embedding_2d[mask, 1],
                    c=[colors[i]], 
                    label=label, 
                    alpha=0.7
                )
            ax.legend()
        else:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7)
        
        ax.set_title(f"Embedding Space Visualization ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot attention weights as a heatmap."""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        sns.heatmap(
            attention_weights,
            ax=ax,
            cmap="Blues",
            xticklabels=tokens if tokens else False,
            yticklabels=tokens if tokens else False,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title("Attention Pattern Heatmap")
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")
        
        if tokens:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_head_specialization(
        self,
        head_catalog: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot attention head specialization scores."""
        
        # Extract data
        syntactic_scores = [h["scores"]["syntactic_score"] for h in head_catalog]
        factual_scores = [h["scores"]["factual_score"] for h in head_catalog]
        functions = [h["function"] for h in head_catalog]
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Color by function
        unique_functions = list(set(functions))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_functions)))
        color_map = {func: colors[i] for i, func in enumerate(unique_functions)}
        
        for func in unique_functions:
            mask = np.array(functions) == func
            ax.scatter(
                np.array(syntactic_scores)[mask],
                np.array(factual_scores)[mask],
                c=[color_map[func]],
                label=func,
                alpha=0.7,
                s=50
            )
        
        ax.set_xlabel("Syntactic Score")
        ax.set_ylabel("Factual Score")
        ax.set_title("Attention Head Specialization")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_layer_analysis(
        self,
        layer_metrics: Dict[str, Dict[str, float]],
        metric_name: str = "variance_explained",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot metrics across layers."""
        
        # Extract layer indices and metrics
        layers = []
        values = []
        
        for layer_key, metrics in layer_metrics.items():
            if layer_key.startswith("layer_"):
                layer_idx = int(layer_key.split("_")[1])
                if metric_name in metrics:
                    layers.append(layer_idx)
                    values.append(metrics[metric_name])
        
        # Sort by layer index
        sorted_data = sorted(zip(layers, values))
        layers, values = zip(*sorted_data) if sorted_data else ([], [])
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(layers, values, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{metric_name.replace('_', ' ').title()} Across Layers")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cluster_analysis(
        self,
        cluster_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot cluster analysis results."""
        
        # Extract cluster information
        cluster_names = list(cluster_data.keys())
        cluster_sizes = [cluster_data[name]["size"] for name in cluster_names]
        cluster_proportions = [cluster_data[name]["proportion"] for name in cluster_names]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cluster sizes bar plot
        ax1.bar(range(len(cluster_names)), cluster_sizes)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Cluster Size")
        ax1.set_title("Cluster Sizes")
        ax1.set_xticks(range(len(cluster_names)))
        ax1.set_xticklabels([f"C{i}" for i in range(len(cluster_names))])
        
        # Cluster proportions pie chart
        ax2.pie(cluster_proportions, labels=[f"C{i}" for i in range(len(cluster_names))], autopct='%1.1f%%')
        ax2.set_title("Cluster Proportions")
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_validation_results(
        self,
        validation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot validation results summary."""
        
        # Extract component scores
        component_scores = validation_results.get("summary", {}).get("component_scores", {})
        
        if not component_scores:
            logger.warning("No component scores found in validation results")
            return plt.figure()
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        components = list(component_scores.keys())
        scores = list(component_scores.values())
        
        # Create bar plot
        bars = ax.bar(range(len(components)), scores)
        
        # Color bars based on score
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score > 0.8:
                bar.set_color('green')
            elif score > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel("Component")
        ax.set_ylabel("Score")
        ax.set_title("Validation Results by Component")
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add overall score as text
        overall_score = validation_results.get("summary", {}).get("overall_score", 0)
        ax.text(0.02, 0.98, f"Overall Score: {overall_score:.3f}", 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_transplant_comparison(
        self,
        baseline_metrics: Dict[str, float],
        transplant_metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison between baseline and transplanted model."""
        
        # Find common metrics
        common_metrics = set(baseline_metrics.keys()) & set(transplant_metrics.keys())
        
        if not common_metrics:
            logger.warning("No common metrics found for comparison")
            return plt.figure()
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        metrics = list(common_metrics)
        baseline_values = [baseline_metrics[m] for m in metrics]
        transplant_values = [transplant_metrics[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create grouped bar plot
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, transplant_values, width, label='Transplanted', alpha=0.8)
        
        # Color bars based on improvement
        for i, (baseline, transplant) in enumerate(zip(baseline_values, transplant_values)):
            if transplant > baseline:
                bars2[i].set_color('green')
            else:
                bars2[i].set_color('red')
        
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_title("Baseline vs Transplanted Model Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_analysis_dashboard(
        self,
        analysis_results: Dict[str, Any],
        output_dir: str
    ) -> None:
        """Create a comprehensive analysis dashboard."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot head specialization if available
        if "head_catalog" in analysis_results:
            fig = self.plot_head_specialization(
                analysis_results["head_catalog"],
                save_path=output_path / "head_specialization.png"
            )
            plt.close(fig)
        
        # Plot layer analysis for FFN metrics
        if "component_analysis" in analysis_results and "ffn_layers" in analysis_results["component_analysis"]:
            ffn_metrics = {}
            for layer_key, layer_data in analysis_results["component_analysis"]["ffn_layers"].items():
                if "metrics" in layer_data:
                    ffn_metrics[layer_key] = layer_data["metrics"]
            
            if ffn_metrics:
                fig = self.plot_layer_analysis(
                    ffn_metrics,
                    "pca_variance_explained",
                    save_path=output_path / "ffn_variance_by_layer.png"
                )
                plt.close(fig)
        
        logger.info(f"Analysis dashboard created in: {output_path}")
    
    def save_all_plots(self, output_dir: str) -> None:
        """Save all currently open plots to output directory."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all figure numbers
        fig_nums = plt.get_fignums()
        
        for i, fig_num in enumerate(fig_nums):
            fig = plt.figure(fig_num)
            fig.savefig(output_path / f"plot_{i+1}.png", dpi=300, bbox_inches='tight')
        
        logger.info(f"Saved {len(fig_nums)} plots to: {output_path}")