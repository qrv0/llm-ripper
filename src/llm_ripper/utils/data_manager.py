"""
Data management utilities for LLM Ripper.
"""

import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading, preprocessing, and storage for LLM Ripper."""
    
    def __init__(self, config):
        self.config = config
        
    def load_probing_corpus(self, corpus_type: str = "diverse") -> Dataset:
        """
        Load a corpus designed for probing specific capabilities.
        
        Args:
            corpus_type: Type of corpus ("diverse", "syntactic", "factual", "semantic")
            
        Returns:
            Dataset containing text samples
        """
        if corpus_type == "diverse":
            return self._create_diverse_corpus()
        elif corpus_type == "syntactic":
            return self._create_syntactic_corpus()
        elif corpus_type == "factual":
            return self._create_factual_corpus()
        elif corpus_type == "semantic":
            return self._create_semantic_corpus()
        else:
            raise ValueError(f"Unknown corpus type: {corpus_type}")
    
    def _create_diverse_corpus(self) -> Dataset:
        """Create a diverse corpus for general probing."""
        
        # Mix of different text types
        texts = [
            # News-style
            "The Federal Reserve announced a new monetary policy decision today.",
            "Scientists have discovered a new exoplanet in a distant solar system.",
            "The stock market experienced significant volatility following the announcement.",
            
            # Academic/Technical
            "Machine learning algorithms require large datasets for effective training.",
            "The transformer architecture has revolutionized natural language processing.",
            "Quantum computing promises to solve previously intractable computational problems.",
            
            # Literature/Creative
            "The old oak tree stood majestically in the moonlit garden.",
            "She wandered through the bustling marketplace, absorbing the vibrant atmosphere.",
            "Time seemed to slow as he contemplated the vast expanse of the ocean.",
            
            # Conversational
            "Can you help me understand how this works?",
            "I think we should consider all the available options.",
            "What do you think about the proposal they presented yesterday?",
            
            # Factual/Encyclopedic
            "Paris is the capital and largest city of France.",
            "The human brain contains approximately 86 billion neurons.",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
        ]
        
        return Dataset.from_dict({"text": texts})
    
    def _create_syntactic_corpus(self) -> Dataset:
        """Create a corpus focused on syntactic structures."""
        
        texts = [
            # Simple sentences
            "The cat sits on the mat.",
            "Birds fly in the sky.",
            "Children play in the park.",
            
            # Complex sentences with dependencies
            "The book that I read yesterday was fascinating.",
            "After the meeting ended, we discussed the proposal.",
            "Although it was raining, they decided to go for a walk.",
            
            # Questions and imperatives
            "What time does the store close?",
            "Please bring me the documents from the office.",
            "How did you solve this difficult problem?",
            
            # Passive constructions
            "The house was built by experienced carpenters.",
            "The problem will be solved by our team.",
            "The letter has been delivered to the recipient.",
            
            # Relative clauses
            "The person who called you is waiting outside.",
            "The movie that we watched last night was excellent.",
            "The place where we met is now a restaurant.",
        ]
        
        return Dataset.from_dict({"text": texts})
    
    def _create_factual_corpus(self) -> Dataset:
        """Create a corpus focused on factual knowledge."""
        
        texts = [
            # Geography
            "Mount Everest is the highest mountain in the world.",
            "The Amazon River flows through South America.",
            "Australia is both a country and a continent.",
            
            # History
            "World War II ended in 1945.",
            "The Berlin Wall fell in 1989.",
            "The Renaissance began in Italy during the 14th century.",
            
            # Science
            "Water boils at 100 degrees Celsius at sea level.",
            "The speed of light is approximately 300,000 kilometers per second.",
            "DNA contains the genetic instructions for all living organisms.",
            
            # Famous people
            "Albert Einstein developed the theory of relativity.",
            "William Shakespeare wrote Romeo and Juliet.",
            "Marie Curie won Nobel Prizes in both Physics and Chemistry.",
            
            # Current events (general knowledge)
            "The Internet has transformed global communication.",
            "Climate change is a significant environmental challenge.",
            "Artificial intelligence is advancing rapidly across many fields.",
        ]
        
        return Dataset.from_dict({"text": texts})
    
    def _create_semantic_corpus(self) -> Dataset:
        """Create a corpus focused on semantic relationships."""
        
        texts = [
            # Synonyms and similar meanings
            "The car is fast. The automobile is quick.",
            "She is happy. She feels joyful and content.",
            "The house is large. The building is spacious.",
            
            # Antonyms and contrasts
            "It's hot outside. The weather is definitely not cold.",
            "He is tall while his brother is short.",
            "The task is easy, unlike the difficult assignment.",
            
            # Hierarchical relationships
            "A rose is a type of flower. Flowers are plants.",
            "Mammals include dogs, cats, and humans.",
            "Vehicles encompass cars, trucks, and motorcycles.",
            
            # Cause and effect
            "Because it rained, the ground became wet.",
            "The loud noise startled the cat, causing it to run away.",
            "Due to the traffic jam, we arrived late to the meeting.",
            
            # Metaphorical language
            "Time is money in the business world.",
            "Her voice was music to his ears.",
            "The classroom was a zoo during the break.",
        ]
        
        return Dataset.from_dict({"text": texts})
    
    def create_validation_dataset(self, task_type: str) -> Dataset:
        """Create validation datasets for specific tasks."""
        
        if task_type == "cola":  # Grammatical acceptability
            return self._create_cola_dataset()
        elif task_type == "stsb":  # Semantic similarity
            return self._create_stsb_dataset()
        elif task_type == "pos":  # Part-of-speech tagging
            return self._create_pos_dataset()
        elif task_type == "ner":  # Named entity recognition
            return self._create_ner_dataset()
        else:
            raise ValueError(f"Unknown validation task: {task_type}")
    
    def _create_cola_dataset(self) -> Dataset:
        """Create a simplified CoLA-style dataset."""
        
        sentences = [
            {"text": "The cat sat on the mat.", "label": 1},
            {"text": "Cat the sat mat on the.", "label": 0},
            {"text": "She quickly ran to the store.", "label": 1},
            {"text": "Quickly she ran store the to.", "label": 0},
            {"text": "The book is on the table.", "label": 1},
            {"text": "Is book the table on the.", "label": 0},
            {"text": "They will arrive tomorrow morning.", "label": 1},
            {"text": "Tomorrow will they morning arrive.", "label": 0},
        ]
        
        texts = [s["text"] for s in sentences]
        labels = [s["label"] for s in sentences]
        
        return Dataset.from_dict({"text": texts, "label": labels})
    
    def _create_stsb_dataset(self) -> Dataset:
        """Create a simplified STS-B style dataset."""
        
        sentence_pairs = [
            {"sentence1": "The cat is sleeping.", "sentence2": "A cat is napping.", "score": 4.5},
            {"sentence1": "I love pizza.", "sentence2": "Pizza is delicious.", "score": 3.8},
            {"sentence1": "The car is red.", "sentence2": "The vehicle is crimson.", "score": 4.2},
            {"sentence1": "He runs fast.", "sentence2": "She walks slowly.", "score": 1.5},
            {"sentence1": "The weather is nice.", "sentence2": "It's a beautiful day.", "score": 4.0},
            {"sentence1": "Dogs are animals.", "sentence2": "Cats are pets.", "score": 2.8},
            {"sentence1": "I'm going home.", "sentence2": "I'm heading to my house.", "score": 4.7},
            {"sentence1": "The book is heavy.", "sentence2": "The music is loud.", "score": 0.5},
        ]
        
        return Dataset.from_dict({
            "sentence1": [p["sentence1"] for p in sentence_pairs],
            "sentence2": [p["sentence2"] for p in sentence_pairs],
            "score": [p["score"] for p in sentence_pairs]
        })
    
    def _create_pos_dataset(self) -> Dataset:
        """Create a simplified POS tagging dataset."""
        
        examples = [
            {
                "tokens": ["The", "quick", "brown", "fox", "jumps"],
                "pos_tags": ["DT", "JJ", "JJ", "NN", "VBZ"]
            },
            {
                "tokens": ["She", "runs", "fast", "every", "morning"],
                "pos_tags": ["PRP", "VBZ", "RB", "DT", "NN"]
            },
            {
                "tokens": ["They", "will", "arrive", "tomorrow", "evening"],
                "pos_tags": ["PRP", "MD", "VB", "NN", "NN"]
            },
        ]
        
        return Dataset.from_dict({
            "tokens": [ex["tokens"] for ex in examples],
            "pos_tags": [ex["pos_tags"] for ex in examples]
        })
    
    def _create_ner_dataset(self) -> Dataset:
        """Create a simplified NER dataset."""
        
        examples = [
            {
                "tokens": ["John", "works", "at", "Microsoft", "in", "Seattle"],
                "ner_tags": ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"]
            },
            {
                "tokens": ["Apple", "Inc.", "was", "founded", "by", "Steve", "Jobs"],
                "ner_tags": ["B-ORG", "I-ORG", "O", "O", "O", "B-PER", "I-PER"]
            },
        ]
        
        return Dataset.from_dict({
            "tokens": [ex["tokens"] for ex in examples],
            "ner_tags": [ex["ner_tags"] for ex in examples]
        })
    
    def save_analysis_results(
        self, 
        results: Dict[str, Any], 
        output_path: Union[str, Path]
    ) -> None:
        """Save analysis results to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
    
    def load_analysis_results(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load analysis results from file."""
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")