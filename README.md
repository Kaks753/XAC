# 🚀 XAC: Explainable Analytics Copilot

**Hey! I'm Stephen Muema, and I built this because I was tired of ML models being black boxes.**

This isn't another generic ML tool. XAC is my answer to a simple question: *"Why did my model make that prediction?"* But it's evolved into something way more powerful - **XAC Pro** - your universal ML explainability assistant that understands EVERYTHING about machine learning.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4%EF%B8%8F-red.svg)]()

---

## 🎯 What Is This?

XAC started as a constrained conversational interface for explaining ML predictions. Now it's evolved into **XAC Pro** - think of it as having a senior ML engineer in your terminal who:

✅ **Explains** your model predictions using SHAP and structured evidence  
✅ **Teaches** you ML concepts from beginner to expert level  
✅ **Answers** literally ANY ML question you throw at it  
✅ **Recommends** which algorithms and metrics to use  
✅ **Troubleshoots** your ML problems (low recall? I got you)  
✅ **Works** 100% locally - no API keys, no cloud, no BS  

### Why I Built This

Look, I love ML. But I hate when models act like fortune tellers - giving you predictions with zero explanation. And I especially hate when I have to Google the same ML concepts over and over because I forgot what "bias-variance tradeoff" means.

So I built XAC. First version was simple - explain predictions using SHAP. But then I thought: *"Why stop there?"* Why not build a system that knows EVERYTHING about ML and can teach it to anyone?

That's XAC Pro.

---

## 🆕 XAC Pro: What's New

### Version 1.0 (Classic XAC)
- ✅ Evidence-based explanations
- ✅ SHAP integration
- ✅ Safety guardrails
- ✅ Intent classification

### Version 2.0 (XAC Pro) - **NEW!**
- ✅ **ML Knowledge Base** - 15+ concepts, 12+ algorithms, 12+ metrics
- ✅ **Universal ML Teacher** - Ask ANY ML question
- ✅ **Multi-level Explanations** - Beginner, Expert, Business
- ✅ **Smart Recommendations** - Which algorithm? Which metric?
- ✅ **Troubleshooting** - "Why is my recall 0%?" → Get answers
- 🔄 **Auto-Discovery** - Coming soon: Analyze notebooks automatically
- 🔄 **CLI Interface** - Coming soon: `xac-pro analyze notebook.ipynb`

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/Kaks753/XAC
cd XAC

# Install dependencies (minimal - no paid APIs!)
pip install -r requirements.txt
```

### Use XAC Classic (v1.0) - Explain Predictions

```python
from src.copilot import ExplainableAnalyticsCopilot, EvidenceBuilder

# Initialize
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

# Ask anything
response = copilot.explain("Why is user 102's score low?", evidence)
print(response.explanation)
```

### Use XAC Pro (v2.0) - Ask ANY ML Question

```python
from xac_pro.core.ml_teacher import MLTeacher

# Initialize ML Teacher
teacher = MLTeacher()

# Ask literally anything about ML
print(teacher.ask("What is overfitting?"))
# → Clear explanation with symptoms and solutions

print(teacher.ask("When should I use XGBoost?"))
# → Detailed guide with strengths, weaknesses, use cases

print(teacher.ask("Why is my recall 0%?"))
# → Troubleshooting guide with 5 common causes + fixes

print(teacher.ask("Which algorithm for small data classification?"))
# → Smart recommendations based on constraints

print(teacher.ask("Difference between precision and recall?"))
# → Side-by-side comparison with examples
```

---

## 💡 What Makes XAC Different?

### 1. Evidence-First, Not Magic
Other tools: *"Trust me bro, this feature is important"*  
XAC: *"Here's the SHAP value (-0.31), feature value (15), and exactly how it contributed"*

### 2. Teaching, Not Just Answering
Other tools: *"Overfitting is when your model..."* (generic definition)  
XAC Pro: *Beginner explanation + symptoms + solutions + related concepts + expert math*

### 3. Context-Aware Explanations
- **Beginner:** "Think of it like sorting mail into boxes..."
- **Expert:** "Formally defined as C_ij = count(y_true=i, y_pred=j)..."
- **Business:** "Shows prediction accuracy: 85% correct, 15% need review"

### 4. Zero Dependencies on Paid Services
No OpenAI API. No Anthropic. No cloud. No credit card. Just Python.

### 5. Built for Real ML Work
This isn't a toy. I use this in my actual ML projects. It handles:
- Classification, regression, clustering
- Any sklearn model, XGBoost, LightGBM
- Imbalanced data, missing values, outliers
- Production constraints (speed, interpretability)

---

## 🧠 XAC Pro Capabilities

### ML Knowledge Base
```python
from xac_pro.core.knowledge_base import MLKnowledgeBase

kb = MLKnowledgeBase()

# Explain any concept
print(kb.explain_concept("overfitting", level="beginner"))
print(kb.explain_concept("overfitting", level="expert"))

# Explain any algorithm
print(kb.explain_algorithm("xgboost", level="business"))

# Explain any metric
print(kb.explain_metric("f1_score", level="beginner"))

# Get recommendations
algos = kb.get_algorithm_for_problem("classification", ["small_data", "interpretable"])
metrics = kb.get_metrics_for_problem("classification", "imbalanced_classes")
```

### Universal ML Teacher
```python
from xac_pro.core.ml_teacher import MLTeacher

teacher = MLTeacher()

# Works with natural language
teacher.ask("What is bias-variance tradeoff?")
teacher.ask("When should I use Random Forest vs XGBoost?")
teacher.ask("Why is accuracy misleading?")
teacher.ask("How does SMOTE work?")
teacher.ask("Which metric for imbalanced data?")
```

### Knowledge Coverage
- **15+ ML Concepts:** Overfitting, underfitting, bias-variance, regularization, cross-validation, feature importance, SHAP, class imbalance, and more
- **12+ Algorithms:** Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, Naive Bayes, Decision Trees, KNN, Neural Networks, K-Means, PCA, and more
- **12+ Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, MSE, RMSE, MAE, R², Log Loss, and more

---

## 📁 Project Structure

```
XAC/
├── src/copilot/          # XAC Classic (v1.0)
│   ├── copilot.py        # Main orchestration
│   ├── intent.py         # Intent classification
│   ├── evidence_builder.py  # Evidence construction
│   ├── guardrails.py     # Safety boundaries
│   └── prompt_templates.py  # Explanation templates
│
├── xac_pro/              # XAC Pro (v2.0) - NEW!
│   ├── core/
│   │   ├── knowledge_base.py   # ML knowledge system
│   │   └── ml_teacher.py       # Universal ML teacher
│   ├── knowledge/
│   │   ├── ml_concepts.json    # Concept encyclopedia
│   │   ├── algorithms_db.json  # Algorithm database
│   │   └── metrics_db.json     # Metrics reference
│   ├── analyzers/        # Coming soon: Auto-discovery
│   ├── explainers/       # Coming soon: Domain-specific explainers
│   ├── interfaces/       # Coming soon: CLI, web, API
│   └── utils/            # Coming soon: SHAP wrappers, viz
│
├── examples/             # Sample inputs/outputs
├── notebooks/            # Interactive demos
├── XAC_PRO_ROADMAP.md   # Full vision & implementation plan
└── README.md            # You are here!
```

---

## 🎓 Real-World Examples

### Example 1: Explain a Prediction
```python
# Classic XAC workflow
copilot = ExplainableAnalyticsCopilot()
response = copilot.explain("Why is this prediction 78?", evidence)

# Output:
# "User 102's productivity score of 78 is below the baseline of 88.
#  The main contributing factors are:
#  1. Task switching (15 switches) → -0.31 impact
#  2. Sleep deficit (2.5 hours) → -0.22 impact
#  3. Meeting hours (4.2 hours) → -0.15 impact"
```

### Example 2: Learn ML Concepts
```python
# XAC Pro teaching mode
teacher = MLTeacher()

# Beginner explanation
print(teacher.ask("What is overfitting?", level="beginner"))
# "When your model memorizes training data instead of learning patterns..."

# Expert explanation  
print(teacher.ask("What is overfitting?", level="expert"))
# "High variance model that captures noise in training data.
#  Formally, model complexity exceeds optimal point on bias-variance curve..."
```

### Example 3: Troubleshoot Problems
```python
# Real problem I had last week
print(teacher.ask("Why is my recall 0% for class A?"))

# XAC Pro Response:
# "Common causes:
#  1. Class imbalance - Try class_weight='balanced'
#  2. Insufficient data - Use SMOTE
#  3. Features don't discriminate - Feature engineering
#  4. Model too simple - Try XGBoost
#  5. Threshold too high - Lower decision threshold
#  
#  Quick fix: model = LogisticRegression(class_weight='balanced')"
```

### Example 4: Choose Algorithms
```python
# I never remember which algorithm to use
recommendations = kb.get_algorithm_for_problem(
    "classification",
    constraints=["small_data", "interpretable"]
)

# Returns: ["logistic_regression", "decision_tree", "naive_bayes"]
# Each with full explanation of strengths/weaknesses
```

---

## 🔬 Design Philosophy

### 1. Evidence-First Architecture
Every claim must trace back to structured data. No hallucinations. No unfounded assertions.

### 2. Safety by Design
Multiple validation layers. Conservative refusals. Explicit about limitations.

### 3. Teaching Over Telling
Don't just give answers. Build understanding. Connect concepts. Show the "why."

### 4. Real-World Practicality
Built for actual ML work, not academic papers. Handles messy data, imbalanced classes, production constraints.

### 5. Free and Local
No API keys. No cloud. No tracking. Your data never leaves your machine.

---

## 🚧 What's Coming Next

I'm actively developing XAC Pro. Here's what's on the roadmap:

### Phase 2 (Next 2 Weeks)
- [ ] **Auto-Discovery Engine** - `xac.analyze("notebook.ipynb")` → Full analysis
- [ ] **Notebook Analyzer** - Parse .ipynb, extract models, metrics, issues
- [ ] **Model Analyzer** - Auto-detect model type, hyperparameters, SHAP values
- [ ] **Smart Suggestions** - Auto-recommend improvements

### Phase 3 (Month 2)
- [ ] **CLI Interface** - `xac-pro analyze notebook.ipynb`
- [ ] **Jupyter Magic** - `%%xac` magic command
- [ ] **Web Dashboard** - Streamlit interface
- [ ] **REST API** - FastAPI endpoints

### Phase 4 (Future)
- [ ] Deep learning support (PyTorch, TensorFlow)
- [ ] Time series explainability
- [ ] Fairness and bias detection
- [ ] Automated report generation (PDF/HTML)
- [ ] VS Code extension
- [ ] Team collaboration features

See [XAC_PRO_ROADMAP.md](XAC_PRO_ROADMAP.md) for the complete vision.

---

## 🤝 Contributing

This is my personal project, but I welcome contributions:
- 🐛 Bug reports
- 💡 Feature suggestions  
- 📝 Documentation improvements
- 🧪 Test cases

**How to contribute:**
1. Open an issue describing the improvement
2. Fork the repo
3. Create a feature branch
4. Submit a PR with tests and docs

---

## 📚 Learn More

### Documentation
- [XAC_PRO_ROADMAP.md](XAC_PRO_ROADMAP.md) - Complete vision and implementation plan
- [examples/](examples/) - Sample inputs and outputs
- [notebooks/](notebooks/) - Interactive demos

### Technical Details
- **No Paid APIs:** Everything runs locally
- **SHAP Integration:** Theoretically sound feature attribution
- **Pattern-Based Intent:** Deterministic, not probabilistic
- **JSON Knowledge Base:** Human-readable, easily extensible
- **Multi-Level Explanations:** Beginner/Expert/Business contexts

---

## 👤 About Me

**Stephen Muema**  
ML Engineer | Data Science Student | Builder of Tools

I'm passionate about making ML more interpretable and accessible. This project showcases my skills in:
- 🧠 ML Explainability (SHAP, feature importance, model interpretation)
- 🏗️ Software Architecture (clean code, SOLID principles, testing)
- 🤖 AI Ethics (safety guardrails, bias detection, transparency)
- 📊 Data Science (scikit-learn, XGBoost, pandas, numpy)
- 💬 Communication (technical writing, teaching, documentation)

**Let's connect:**
- 📧 Email: musyokas753@gmail.com
- 💼 LinkedIn: [stephen-muema-617339359](https://www.linkedin.com/in/stephen-muema-617339359)
- 🌐 Portfolio: [stephenmueama.com](https://stephenmueama.com)
- 📍 Location: Kiambu, Kenya

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

TL;DR: Use it, modify it, build on it. Just give credit.

---

## 🌟 Show Your Support

If XAC helped you understand your models better, or saved you time Googling ML concepts:

⭐ **Star this repo**  
🐦 **Share it on Twitter**  
💬 **Tell your ML friends**  
🔗 **Link it in your projects**

Your support means a lot and motivates me to keep building cool ML tools.

---

## 🙏 Acknowledgments

**Inspiration:**
- SHAP library by Scott Lundberg
- Interpretable ML book by Christoph Molnar
- Microsoft's Responsible AI guidelines
- Every ML engineer who's been frustrated by black box models

**Tech Stack:**
- Python 3.12+
- scikit-learn (model integration)
- SHAP (feature attribution)
- NumPy (numerical computing)
- JSON (knowledge storage)

---

**Built with ❤️ and lots of ☕ by Stephen Muema**

*Making ML interpretable, one explanation at a time.*

---

*Last updated: January 14, 2026*
