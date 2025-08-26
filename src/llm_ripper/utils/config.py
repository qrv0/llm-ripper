"""Configuration management for LLM Ripper."""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


class ConfigManager:
    """Manages configuration settings for the LLM Ripper framework."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment variables."""
        # Load .env if present
        env_path = Path(self.config_path).parent if self.config_path else Path(".")
        dot_env = env_path / ".env"
        if dot_env.exists():
            load_dotenv(dotenv_path=dot_env)
        config = {
            # Model Configuration
            # Require explicit model identifiers (no defaults)
            "donor_model_name": os.getenv("DONOR_MODEL_NAME"),
            "target_model_name": os.getenv("TARGET_MODEL_NAME"),
            "model_cache_dir": os.getenv("MODEL_CACHE_DIR", "./models"),
            "device": os.getenv("DEVICE", "auto"),  # auto|cuda|cpu|mps
            # Data Configuration
            "knowledge_bank_dir": os.getenv("KNOWLEDGE_BANK_DIR", "./knowledge_bank"),
            "corpus_dir": os.getenv("CORPUS_DIR", "./corpus"),
            "output_dir": os.getenv("OUTPUT_DIR", "./output"),
            "hdf5_compression": os.getenv("HDF5_COMPRESSION", "gzip"),
            "hdf5_store_mode": os.getenv(
                "HDF5_STORE_MODE", "groups"
            ),  # groups|dataset (experimental)
            "offline": os.getenv("OFFLINE", "false").lower() == "true",
            "seed": int(os.getenv("SEED", "42")),
            # Extraction Configuration
            "batch_size": int(os.getenv("BATCH_SIZE", "8")),
            "max_sequence_length": int(os.getenv("MAX_SEQUENCE_LENGTH", "512")),
            "use_safetensors": os.getenv("USE_SAFETENSORS", "true").lower() == "true",
            "capture_all_layers": os.getenv("CAPTURE_ALL_LAYERS", "true").lower()
            == "true",
            "enable_downstream_perplexity": os.getenv(
                "ENABLE_DOWNSTREAM_PERPLEXITY", "false"
            ).lower()
            == "true",
            "tensor_shard_mb": int(os.getenv("TENSOR_SHARD_MB", "512")),
            # Analysis Configuration
            "pca_components": int(os.getenv("PCA_COMPONENTS", "50")),
            "clustering_algorithm": os.getenv("CLUSTERING_ALGORITHM", "kmeans"),
            "n_clusters": int(os.getenv("N_CLUSTERS", "10")),
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.8")),
            "store_attn_per_head": os.getenv("STORE_ATTN_PER_HEAD", "false").lower()
            == "true",
            # Transplant Configuration
            "adapter_hidden_size": int(os.getenv("ADAPTER_HIDDEN_SIZE", "64")),
            "freeze_donor_weights": os.getenv("FREEZE_DONOR_WEIGHTS", "true").lower()
            == "true",
            "learning_rate": float(os.getenv("LEARNING_RATE", "1e-4")),
            "num_epochs": int(os.getenv("NUM_EPOCHS", "3")),
            # Validation Configuration
            "validation_datasets": os.getenv(
                "VALIDATION_DATASETS", "cola,stsb,mnli"
            ).split(","),
            "validation_batch_size": int(os.getenv("VALIDATION_BATCH_SIZE", "16")),
            "num_validation_samples": int(os.getenv("NUM_VALIDATION_SAMPLES", "1000")),
            # Logging Configuration
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "wandb_project": os.getenv("WANDB_PROJECT", "llm-ripper"),
            "wandb_entity": os.getenv("WANDB_ENTITY", None),
            "save_checkpoints": os.getenv("SAVE_CHECKPOINTS", "true").lower() == "true",
            # Hardware Configuration
            "num_workers": int(os.getenv("NUM_WORKERS", "4")),
            "pin_memory": os.getenv("PIN_MEMORY", "true").lower() == "true",
            "mixed_precision": os.getenv("MIXED_PRECISION", "true").lower() == "true",
            # Loader/quantization options
            "load_in_8bit": os.getenv("LOAD_IN_8BIT", "false").lower() == "true",
            "load_in_4bit": os.getenv("LOAD_IN_4BIT", "false").lower() == "true",
            "trust_remote_code": os.getenv("TRUST_REMOTE_CODE", "false").lower()
            == "true",
            "attn_implementation": os.getenv("ATTN_IMPLEMENTATION", None),
            # Rope scaling (JSON string or literal string); applied best-effort in ModelLoader
            "rope_scaling": os.getenv("ROPE_SCALING", None),
            "compile_model": os.getenv("COMPILE_MODEL", "false").lower() == "true",
            "use_model_outputs": os.getenv("USE_MODEL_OUTPUTS", "false").lower()
            == "true",
        }

        # Load from config file if provided
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, "r") as f:
                file_config = json.load(f)
                config.update(file_config)

        return config

    # Context-aware validations for CLI flows
    def validate_for_extract(
        self, model_name: Optional[str], output_dir: Optional[str]
    ) -> bool:
        if not model_name:
            raise ValueError("Parameter --model is required for 'extract'.")
        if not output_dir:
            raise ValueError("Parameter --output-dir is required for 'extract'.")
        return True

    def validate_for_capture(
        self, model_name: Optional[str], output_file: Optional[str]
    ) -> bool:
        if not model_name:
            raise ValueError("Parameter --model is required for 'capture'.")
        if not output_file:
            raise ValueError("Parameter --output-file is required for 'capture'.")
        return True

    def validate_for_analyze(
        self, knowledge_bank: Optional[str], output_dir: Optional[str]
    ) -> bool:
        if not knowledge_bank:
            raise ValueError("Parameter --knowledge-bank is required for 'analyze'.")
        if not output_dir:
            raise ValueError("Parameter --output-dir is required for 'analyze'.")
        return True

    def validate_for_transplant(
        self, source: Optional[str], target: Optional[str], output_dir: Optional[str]
    ) -> bool:
        if not source:
            raise ValueError("Parameter --source is required for 'transplant'.")
        if not target:
            raise ValueError("Parameter --target is required for 'transplant'.")
        if not output_dir:
            raise ValueError("Parameter --output-dir is required for 'transplant'.")
        return True

    def validate_for_validate(
        self, model: Optional[str], output_dir: Optional[str]
    ) -> bool:
        if not model:
            raise ValueError("Parameter --model is required for 'validate'.")
        if not output_dir:
            raise ValueError("Parameter --output-dir is required for 'validate'.")
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self.config.update(updates)

    def save(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)

    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        dirs_to_create = [
            self.get("model_cache_dir"),
            self.get("knowledge_bank_dir"),
            self.get("corpus_dir"),
            self.get("output_dir"),
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def validate_config(self) -> bool:
        """Validate configuration settings."""
        required_keys = [
            "donor_model_name",
            "target_model_name",
            "knowledge_bank_dir",
            "output_dir",
        ]

        for key in required_keys:
            if not self.get(key):
                # Provide actionable guidance for missing model names
                if key in ("donor_model_name", "target_model_name"):
                    raise ValueError(
                        f"Missing config '{key}'. Set via config file or environment variables: "
                        f"DONOR_MODEL_NAME and TARGET_MODEL_NAME. Empty values are not accepted."
                    )
                raise ValueError(
                    f"Required configuration key '{key}' is missing or empty"
                )

        # Validate numeric values
        if self.get("batch_size") <= 0:
            raise ValueError("batch_size must be positive")

        if self.get("max_sequence_length") <= 0:
            raise ValueError("max_sequence_length must be positive")

        if self.get("learning_rate") <= 0:
            raise ValueError("learning_rate must be positive")

        return True
