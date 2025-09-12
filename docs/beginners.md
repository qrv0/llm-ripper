# 👶 Beginner's Guide

Welcome to LLM Ripper! This guide is perfect if you're:

- 🆕 New to machine learning
- 🐍 Getting started with Python ML projects  
- 🤖 Curious about how language models work
- 🔬 Want to try model analysis without complex setup

!!! tip "No GPU Required!"
    
    Everything in this guide works on your CPU - perfect for laptops and getting started!

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:

- ✅ Install and run LLM Ripper
- ✅ Analyze a language model's attention patterns
- ✅ Understand what different components do
- ✅ Generate your first safety report
- ✅ Use the visual Studio interface

## 🛠️ Prerequisites  

Make sure you have:

- **Python 3.8 or newer** ([Download here](https://python.org/downloads/))
- **5 minutes of your time** ⏰

??? question "How to check your Python version"
    
    ```bash
    python --version
    # Should show Python 3.8.x or higher
    ```

## 📦 Step 1: Installation

The easiest way to get started:

```bash
pip install llm-ripper
```

!!! success "That's it!"
    
    You now have LLM Ripper installed! Let's verify it worked:
    
    ```bash
    llm-ripper --help
    ```

??? info "Advanced: Development Setup"
    
    If you want to contribute or modify LLM Ripper:
    
    ```bash
    # Clone the repository
    git clone https://github.com/qrv0/LLM-Ripper.git
    cd LLM-Ripper
    
    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    
    # Install in development mode
    pip install -e .
    ```

## 🚀 Step 2: Your First Analysis

Let's start with something simple - analyzing how a small language model processes text:

### Create Your First Script

Create a new file called `my_first_analysis.py`:

```python title="my_first_analysis.py"
from llm_ripper import LLMRipper

# We'll use a small, fast model perfect for learning
print("🔄 Loading model (this may take a moment)...")
ripper = LLMRipper("distilgpt2")  # Small but capable model
print("✅ Model loaded!")

# Let's analyze how it processes a simple sentence
text = "The cat sat on the mat and looked around."
print(f"📝 Analyzing text: '{text}'")

# Extract attention patterns - this shows what words the model focuses on
print("🔍 Extracting attention patterns...")
attention_data = ripper.extract_attention_patterns(
    text=text,
    layers=[0, 1, 2]  # Look at the first 3 layers
)

print(f"✅ Found attention patterns in {len(attention_data)} layers")

# Let's see what components are most important
print("📊 Analyzing component importance...")
importance = ripper.analyze_importance(
    method="attention",  # Use attention weights to measure importance
    target_layer=2  # Focus on layer 2
)

print("🎉 Analysis complete!")
print(f"📈 Found {len(importance.components)} important components")

# Show the top 5 most important components
top_5 = importance.get_top_k(k=5)
print("\n🏆 Top 5 Most Important Components:")
for i, comp in enumerate(top_5, 1):
    print(f"  {i}. {comp.name}: {comp.score:.3f}")
```

### Run Your Analysis

```bash
python my_first_analysis.py
```

You should see output like:

```
🔄 Loading model (this may take a moment)...
✅ Model loaded!
📝 Analyzing text: 'The cat sat on the mat and looked around.'
🔍 Extracting attention patterns...
✅ Found attention patterns in 3 layers
📊 Analyzing component importance...
🎉 Analysis complete!
📈 Found 144 important components

🏆 Top 5 Most Important Components:
  1. attention_head_0_2: 0.847
  2. attention_head_1_5: 0.793
  3. attention_head_2_1: 0.761
  4. ffn_layer_1: 0.725
  5. attention_head_0_7: 0.698
```

!!! success "Congratulations! 🎉"
    
    You just performed your first language model analysis! Those numbers show which parts of the model were most active when processing your text.

## 🎨 Step 3: Visual Exploration

Now let's use the Studio interface to explore your results visually:

```bash
# Generate some analysis results first
llm-ripper analyze --model distilgpt2 --output ./my_analysis --quick

# Launch the visual interface
llm-ripper studio --results ./my_analysis --port 8000
```

Then open your browser to `http://localhost:8000` to see:

- 🔍 **Interactive attention visualizations**
- 📊 **Component importance charts**
- 🎯 **Layer-by-layer breakdowns**
- 📁 **All your generated files**

## 🛡️ Step 4: Safety Analysis

Let's check how safe and reliable our model is:

```python title="safety_check.py"
from llm_ripper import LLMRipper

ripper = LLMRipper("distilgpt2")

print("🛡️ Running safety analysis...")

# Generate a comprehensive safety report
safety_report = ripper.generate_safety_report(
    test_inputs=[
        "Tell me about cats",
        "What is the weather like?", 
        "How do I make a sandwich?"
    ],
    metrics=["consistency", "bias_detection", "output_quality"]
)

print(f"📊 Overall safety score: {safety_report.overall_score}/10")
print(f"🎯 Consistency score: {safety_report.consistency_score}/10") 
print(f"⚖️ Bias detection score: {safety_report.bias_score}/10")

if safety_report.overall_score >= 7:
    print("✅ Model appears to be operating safely!")
elif safety_report.overall_score >= 5:
    print("⚠️ Model shows some concerning patterns")
else:
    print("🚨 Model may need additional safety measures")
```

## 🔄 Step 5: Knowledge Transfer (Advanced)

Ready for something more advanced? Let's try transferring knowledge between models:

```python title="knowledge_transfer.py"
from llm_ripper import LLMRipper

# Load source model (where we extract knowledge from)
print("📚 Loading source model...")
source = LLMRipper("gpt2")

# Load target model (where we transfer knowledge to) 
print("🎯 Loading target model...")
target = LLMRipper("distilgpt2")

print("🔄 Extracting knowledge from source...")
# Extract specific types of knowledge
knowledge = source.extract_knowledge(
    components=["attention_heads", "layer_norms"],
    layers=range(0, 3),  # First 3 layers only
    knowledge_types=["linguistic", "factual"]
)

print(f"📦 Extracted {len(knowledge.components)} knowledge components")

print("🚚 Transplanting knowledge to target...")
# Transfer with safety checks
result = target.transplant_knowledge(
    knowledge=knowledge,
    safety_threshold=0.8,  # High safety requirement
    validate=True  # Double-check everything works
)

if result.success:
    print("✅ Knowledge transfer successful!")
    print(f"🎯 Transfer accuracy: {result.accuracy:.2%}")
    print(f"🛡️ Safety maintained: {result.safety_preserved}")
else:
    print("❌ Transfer failed:")
    print(f"   Reason: {result.error_message}")
```

## 🎓 Understanding the Results

### What Are Attention Patterns?

Think of attention patterns like **focus maps** - they show which words the model is "paying attention to" when processing each word:

- **High attention** = "This word is very important for understanding"  
- **Low attention** = "This word isn't crucial right now"

### What Are Components?

Language models are made of many **components** that each do different jobs:

- **🎯 Attention Heads**: Focus on relationships between words
- **🧠 Feed-Forward Networks (FFNs)**: Process and transform information
- **📏 Layer Norms**: Keep the math stable and organized

### What Does Importance Mean?

**Component importance** tells you which parts of the model are doing the heavy lifting:

- **High importance** = This component significantly affects the output
- **Low importance** = This component could probably be removed without much impact

## ❓ Common Questions

??? question "Why is my analysis slow?"

    - **First run**: Models need to be downloaded (happens once)
    - **Large models**: Try smaller models like `distilgpt2` first
    - **No GPU**: CPU analysis is slower but works fine for learning

??? question "Can I analyze my own models?"

    Yes! LLM Ripper works with:
    - ✅ Models from Hugging Face Hub
    - ✅ Your locally fine-tuned models
    - ✅ Most transformer architectures (GPT, BERT, T5, etc.)

??? question "What if I get an error?"

    Common solutions:
    
    1. **Update LLM Ripper**: `pip install --upgrade llm-ripper`
    2. **Check Python version**: Needs 3.8 or newer
    3. **Memory issues**: Try smaller models or texts
    4. **Internet required**: For downloading models first time

??? question "Is this safe to run?"

    Yes! LLM Ripper:
    - ✅ Only reads model weights (doesn't modify them)
    - ✅ Runs locally on your machine
    - ✅ Doesn't send data anywhere
    - ✅ Uses standard, well-tested libraries

## 🎯 What's Next?

Congratulations! You've learned the basics of LLM Ripper. Here's what to explore next:

<div class="grid cards" markdown>

-   **🚀 Ready for More?**
    
    Try the [Quick Start Guide](quickstart.md) for more advanced features

-   **🏗️ How It Works**
    
    Learn about the [Architecture](architecture.md) behind LLM Ripper

-   **📚 Real Examples**
    
    Explore comprehensive [User Guides](guides/end_to_end.md)

-   **🔧 Full Reference**
    
    Browse the complete [API Documentation](api.md)

</div>

## 🆘 Need Help?

- 🐛 **Found a bug?** [Report it on GitHub](https://github.com/qrv0/LLM-Ripper/issues)
- 💬 **Have questions?** [Join our discussions](https://github.com/qrv0/LLM-Ripper/discussions)  
- 📧 **Want to chat?** Email [qorvuscompany@gmail.com](mailto:qorvuscompany@gmail.com)

---

<div align="center" markdown="1">

**Feeling confident? Ready for the next level?**

[Explore Advanced Features :material-rocket-launch:](quickstart.md){ .md-button .md-button--primary }

</div>
- Launch the Studio viewer at http://localhost:8000

If the page shows empty panels or error messages, that's okay — you can still explore the layout and JSON files.

## 3) Next steps
- Try the offline smoke test: `make smoke-offline`
- Explore CLI help: `python -m llm_ripper.cli --help`
- Read the Quickstart for full pipeline steps

## Troubleshooting
- If `mkdocs` or `ruff` commands are missing, install dev deps: `pip install -r requirements-dev.txt`
- If ports are busy, change the Studio port: `make studio PORT=8001`

You're set! As you gain confidence, switch from the demo to actual models and data.
