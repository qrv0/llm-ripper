def test_imports():
    import llm_ripper
    assert hasattr(llm_ripper, "KnowledgeExtractor")
    assert hasattr(llm_ripper, "KnowledgeAnalyzer")
    assert hasattr(llm_ripper, "KnowledgeTransplanter")
    assert hasattr(llm_ripper, "ValidationSuite")
