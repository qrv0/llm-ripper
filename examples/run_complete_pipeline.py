#!/usr/bin/env python3
"""
Pipeline completo de uso do LLM Ripper.

Este script executa o fluxo ponta a ponta:
1. Extração de conhecimento do modelo doador
2. Captura de ativações em um corpus
3. Análise dos componentes extraídos
4. Transplante dos componentes ao modelo alvo
5. Validação do modelo transplantado

Observação: nomes de modelos não são definidos aqui por intenção do projeto.
Forneça-os via arquivo de configuração ou variáveis de ambiente
(`DONOR_MODEL_NAME`, `TARGET_MODEL_NAME`).
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_ripper.utils.config import ConfigManager
from llm_ripper.core.extraction import KnowledgeExtractor
from llm_ripper.core.activation_capture import ActivationCapture
from llm_ripper.core.analysis import KnowledgeAnalyzer
from llm_ripper.core.transplant import KnowledgeTransplanter, TransplantConfig
from llm_ripper.core.validation import ValidationSuite
from llm_ripper.utils.data_manager import DataManager


def main():
    """Run the complete LLM Ripper pipeline."""
    
    print("🧠 LLM Ripper - Pipeline Completo")
    print("=" * 50)
    
    # Load configuration
    config = ConfigManager("examples/config.json")
    config.validate_config()
    config.create_directories()
    
    # Initialize components
    extractor = KnowledgeExtractor(config)
    capture = ActivationCapture(config)
    analyzer = KnowledgeAnalyzer(config)
    transplanter = KnowledgeTransplanter(config)
    validator = ValidationSuite(config)
    data_manager = DataManager(config)
    
    # Step 1: Extract knowledge from donor model
    print("\n1️⃣  Extracting knowledge from donor model...")
    donor_model = config.get("donor_model_name")
    knowledge_bank_dir = config.get("knowledge_bank_dir")
    
    extraction_result = extractor.extract_model_components(
        model_name=donor_model,
        output_dir=knowledge_bank_dir,
        components=["embeddings", "attention_heads", "ffn_layers", "lm_head"]
    )
    
    print(f"✓ Extracted components: {list(extraction_result['extracted_components'].keys())}")
    
    # Step 2: Capture activations
    print("\n2️⃣  Capturing activations...")
    corpus = data_manager.load_probing_corpus("diverse")
    activations_file = str(Path(config.get("output_dir")) / "activations.h5")
    
    capture_result = capture.capture_model_activations(
        model_name=donor_model,
        corpus_dataset=corpus,
        output_file=activations_file,
        max_samples=50  # Small number for example
    )
    
    print(f"✓ Captured activations from {capture_result['num_samples']} samples")
    
    # Step 3: Analyze extracted components
    print("\n3️⃣  Analyzing extracted components...")
    analysis_dir = str(Path(config.get("output_dir")) / "analysis")
    
    analysis_result = analyzer.analyze_knowledge_bank(
        knowledge_bank_dir=knowledge_bank_dir,
        activations_file=activations_file,
        output_dir=analysis_dir
    )
    
    print(f"✓ Analysis completed. Head catalog: {len(analysis_result.get('head_catalog', []))} entries")
    
    # Step 4: Transplant components
    print("\n4️⃣  Transplanting components to target model...")
    target_model = config.get("target_model_name")
    transplant_dir = str(Path(config.get("output_dir")) / "transplanted")
    
    # Define transplant configurations
    transplant_configs = [
        TransplantConfig(
            source_component="layer_2_attention",
            target_layer=1,
            bridge_hidden_size=64,
            freeze_donor=True,
            freeze_target=False,
            strategy="module_injection"
        ),
        TransplantConfig(
            source_component="embeddings",
            target_layer=0,
            bridge_hidden_size=32,
            freeze_donor=True,
            freeze_target=False,
            strategy="embedding_init"
        )
    ]
    
    transplant_result = transplanter.transplant_knowledge(
        source_knowledge_bank=knowledge_bank_dir,
        target_model_name=target_model,
        transplant_configs=transplant_configs,
        output_dir=transplant_dir
    )
    
    print(f"✓ Transplanted {len(transplant_result['transplanted_components'])} components")
    
    # Step 5: Validate transplanted model
    print("\n5️⃣  Validating transplanted model...")
    validation_dir = str(Path(config.get("output_dir")) / "validation")
    
    validation_result = validator.validate_transplanted_model(
        transplanted_model_path=transplant_dir,
        baseline_model_name=target_model,
        benchmarks=["intrinsic", "pos_tagging", "semantic_similarity"],
        output_dir=validation_dir
    )
    
    overall_score = validation_result["summary"]["overall_score"]
    print(f"✓ Validation completed. Overall score: {overall_score:.3f}")
    
    # Print recommendations
    recommendations = validation_result["summary"]["recommendations"]
    if recommendations:
        print("\n📋 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Results saved in:")
    print(f"   • Knowledge bank: {knowledge_bank_dir}")
    print(f"   • Analysis: {analysis_dir}")
    print(f"   • Transplanted model: {transplant_dir}")
    print(f"   • Validation: {validation_dir}")


if __name__ == "__main__":
    main()
