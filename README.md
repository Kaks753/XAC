# 🧠 Explainable Analytics Copilot (XAC)

**A constrained conversational interface for explaining machine learning model outputs using structured evidence, SHAP values, and ethical AI practices.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: professional](https://img.shields.io/badge/code%20style-professional-green.svg)]()

---

## 👤 Author Information

**Muema Stephen**  
Data Science Student | Machine Learning Specialist | Analytics Expert

- 📧 **Email:** musyokas753@gmail.com
- 💼 **LinkedIn:** [www.linkedin.com/in/stephen-muema-617339359](https://www.linkedin.com/in/stephen-muema-617339359)
- 🌐 **Portfolio:** [stephenmueama.com](https://stephenmueama.com)
- 📍 **Location:** Kiambu, Kenya

---

## 🎯 Project Overview

The **Explainable Analytics Copilot (XAC)** is a production-grade system that explains machine learning model predictions using structured, verifiable evidence. Unlike general-purpose AI assistants, XAC is specifically constrained to:

✅ **Explain** existing model outputs  
✅ **Ground** all claims in structured evidence  
✅ **Refuse** predictions, advice, and causal claims  
✅ **Provide** confidence levels and limitations  
✅ **Work** completely free and locally  

### Key Features

- **Evidence-Based Explanations:** All claims traceable to structured JSON evidence
- **SHAP Integration:** Uses SHAP values for theoretically sound feature importance
- **Safety Guardrails:** Multiple validation layers prevent harmful outputs
- **Intent Classification:** Automatically understands user questions
- **Professional Output:** Business-ready explanations with clear structure
- **Free & Local:** No paid APIs, cloud services, or internet required
- **Reusable:** Works across any ML project with predictions and feature importance

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/explainable-analytics-copilot.git
cd explainable-analytics-copilot

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from copilot import (
    ExplainableAnalyticsCopilot,
    EvidenceBuilder,
)

# Initialize copilot
copilot = ExplainableAnalyticsCopilot()

# Build evidence from your model
evidence = EvidenceBuilder.build_user_explanation_evidence(
    entity_id="user_102",
    prediction_value=78.5,
    shap_values={
        'task_switching': -0.31,
        'sleep_deficit': -0.22,
        'meeting_hours': -0.15,
    },
    feature_values={
        'task_switching': 15,
        'sleep_deficit': 2.5,
        'meeting_hours': 4.2,
    },
    base_value=88.0,
)

# Ask a question
response = copilot.explain(
    "Why is user 102's productivity score 78?",
    evidence
)

print(response.explanation)
```

### Interactive Demo

Run the comprehensive Jupyter notebook demo:

```bash
cd notebooks
jupyter notebook demo.ipynb
```

---

## 📁 Project Structure

```
explainable-analytics-copilot/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── src/
│   └── copilot/
│       ├── __init__.py           # Package initialization
│       ├── intent.py             # Intent classification
│       ├── evidence_builder.py   # Evidence construction
│       ├── guardrails.py         # Safety boundaries
│       ├── prompt_templates.py   # Explanation templates
│       └── copilot.py            # Main orchestration
├── examples/
│   ├── example_inputs.json       # Sample queries
│   ├── example_evidence.json     # Sample evidence
│   └── example_outputs.md        # Sample explanations
├── notebooks/
│   └── demo.ipynb                # Interactive demonstration
└── tests/
    └── (unit tests - optional)
```

---

## 🏗️ Architecture

XAC follows a clean pipeline architecture:

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       ↓
┌─────────────────────┐
│ Intent Classifier   │  ← Pattern-based classification
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Guardrails Check    │  ← Refusal patterns, ethical boundaries
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Evidence Retrieval  │  ← Structured JSON with SHAP, metrics
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Evidence Validation │  ← Sufficiency checks
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Template Selection  │  ← Match template to intent
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Explanation Gen.    │  ← Evidence → natural language
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Response Validation │  ← Check for predictions, advice
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Return to User      │
└─────────────────────┘
```

---

## 🧩 Core Components

### 1. Intent Classifier (`intent.py`)

Classifies user queries into supported categories:

- **user_explanation:** Explain a specific prediction
- **feature_importance:** Most important features
- **trend_comparison:** Compare to historical patterns
- **model_performance:** Model quality metrics
- **unsupported_request:** Refused (predictions, advice, etc.)

**Design Choice:** Pattern-based for transparency and determinism.

### 2. Evidence Builder (`evidence_builder.py`)

Constructs structured evidence packages containing:

- Predictions and metadata
- SHAP values / feature contributions
- Historical comparisons
- Model performance metrics
- Confidence levels
- Explicit limitations

**Design Choice:** Structured JSON for reproducibility and auditability.

### 3. Guardrails (`guardrails.py`)

Enforces safety boundaries:

- ❌ No predictions or forecasts
- ❌ No advice or recommendations
- ❌ No causal claims
- ❌ No counterfactual speculation
- ❌ No medical/legal/financial advice

**Design Choice:** Multiple validation layers for defense-in-depth.

### 4. Prompt Templates (`prompt_templates.py`)

Converts evidence to natural language:

- Professional, business-appropriate tone
- Structured, scannable format
- Always includes confidence and limitations
- Markdown formatting for readability

**Design Choice:** Template-based for consistency and control.

### 5. Copilot Orchestrator (`copilot.py`)

Brings everything together:

- Pipeline orchestration
- Error handling
- Logging for debugging
- Structured response objects

**Design Choice:** Clean API, easy integration.

---

## 🎓 Design Rationale

### Why This Approach?

1. **Evidence-First Architecture**
   - Every claim must be traceable to structured data
   - Prevents hallucinations and unfounded assertions
   - Enables audit trails for compliance

2. **Safety by Design**
   - Multiple validation layers (query → evidence → response)
   - Conservative refusal policy (when uncertain, refuse)
   - Explicit about limitations and confidence

3. **Deterministic Behavior**
   - Same query + evidence → same explanation
   - No randomness, no prompt engineering brittleness
   - Fully reproducible for testing and validation

4. **Free and Local**
   - No dependency on paid APIs (OpenAI, Anthropic, etc.)
   - Runs entirely offline
   - No data leaves your infrastructure
   - Portfolio-friendly (anyone can run it)

5. **Production-Ready**
   - Clean architecture with separation of concerns
   - Type hints and docstrings throughout
   - Logging for observability
   - Extensible for new intent types

---

## 📊 Example Use Cases

### 1. Employee Productivity

```python
# Explain why an employee's productivity score is low
query = "Why is user 102's productivity score 78?"
# → Shows task switching, sleep deficit, meeting hours as top factors
```

### 2. Customer Churn

```python
# Identify most important churn predictors
query = "What are the most important features in the churn model?"
# → Ranks features like engagement, tenure, support tickets
```

### 3. Sales Forecasting

```python
# Compare current performance to historical trends
query = "How does this quarter's performance compare to baseline?"
# → Shows deviation from historical average with confidence
```

---

## 🔒 Ethical AI Principles

XAC is designed with ethical AI principles at its core:

1. **Transparency:** Always shows confidence and limitations
2. **Accountability:** All claims traceable to evidence
3. **Safety:** Refuses inappropriate requests
4. **Fairness:** No discrimination in refusal patterns
5. **Privacy:** No data storage or external transmission

---

## 🧪 Testing

### Run Unit Tests

```bash
# If you implement tests (optional but recommended)
pytest tests/ -v --cov=src/copilot
```

### Manual Testing

```bash
# Test intent classification
cd src/copilot
python intent.py

# Test evidence building
python evidence_builder.py

# Test guardrails
python guardrails.py

# Test full copilot
python copilot.py
```

---

## 🚧 Future Enhancements

Potential improvements (without breaking core principles):

- [ ] More sophisticated intent classification (embeddings)
- [ ] Support for more model types (ranking, clustering)
- [ ] Visualization generation (SHAP plots, trends)
- [ ] Multi-language support
- [ ] Confidence calibration using historical accuracy
- [ ] A/B test explanation effectiveness
- [ ] Integration with popular ML frameworks (scikit-learn, XGBoost)

---

## 📚 References & Inspiration

- **SHAP (SHapley Additive exPlanations):** Lundberg & Lee (2017)
- **Interpretable ML:** Molnar, Christoph (2022)
- **Responsible AI:** Microsoft RAI Guidelines
- **Human-AI Interaction:** Amershi et al. (2019)

---

## 🤝 Contributing

While this is primarily a portfolio project, suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💬 Contact & Support

**Muema Stephen**

- 📧 Email: musyokas753@gmail.com
- 💼 LinkedIn: [stephen-muema-617339359](https://www.linkedin.com/in/stephen-muema-617339359)
- 🌐 Portfolio: [stephenmueama.com](https://stephenmueama.com)

For questions, feedback, or collaboration opportunities, feel free to reach out!

---

## 🌟 Acknowledgments

This project demonstrates:

- **ML Interpretability:** SHAP, feature importance, confidence estimation
- **Software Engineering:** Clean architecture, SOLID principles, type safety
- **Ethical AI:** Safety guardrails, transparency, limitations
- **Production Thinking:** Logging, error handling, testing readiness
- **Communication:** Documentation, examples, professional presentation

Built as a portfolio piece showcasing **senior-level data science and ML engineering skills**.

---

**⭐ If you found this project useful, please consider starring the repository!**

---

*Last updated: January 13, 2026*
