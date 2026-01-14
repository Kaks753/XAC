"""
XAC Pro Demo - Showcase the new capabilities

This demonstrates the transformation from XAC Classic to XAC Pro.

Author: Stephen Muema
"""

import sys
sys.path.insert(0, '.')

print("=" * 80)
print("🚀 XAC PRO DEMO - Universal ML Explainability Assistant")
print("=" * 80)
print()

# ============================================================================
# PART 1: XAC Classic (v1.0) - Explain Predictions
# ============================================================================
print("### PART 1: XAC Classic - Explain Model Predictions")
print("-" * 80)

from src.copilot import ExplainableAnalyticsCopilot, EvidenceBuilder

copilot = ExplainableAnalyticsCopilot()

# Build evidence
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

response = copilot.explain("Why is user 102's productivity score 78?", evidence)
print(response.explanation)
print()

# ============================================================================
# PART 2: XAC Pro - Universal ML Teacher
# ============================================================================
print("\n" + "=" * 80)
print("### PART 2: XAC Pro - Ask ANY ML Question")
print("=" * 80)
print()

from xac_pro.core.ml_teacher import MLTeacher

teacher = MLTeacher()

# Example 1: What is?
print("📚 Example 1: What is overfitting?")
print("-" * 80)
answer = teacher.ask("What is overfitting?", level="beginner")
print(answer)
print()

# Example 2: When to use?
print("🤔 Example 2: When should I use XGBoost?")
print("-" * 80)
answer = teacher.ask("When should I use XGBoost?", level="business")
print(answer[:500] + "...\n")  # Truncate for demo

# Example 3: Troubleshooting
print("🔧 Example 3: Why is my recall 0%?")
print("-" * 80)
answer = teacher.ask("Why is my recall 0%?", level="beginner")
print(answer[:600] + "...\n")  # Truncate for demo

# Example 4: Algorithm recommendations
print("💡 Example 4: Which algorithm for small data classification?")
print("-" * 80)
answer = teacher.ask("Which algorithm for small data classification?", level="beginner")
print(answer[:500] + "...\n")  # Truncate for demo

# ============================================================================
# PART 3: XAC Pro - Knowledge Base
# ============================================================================
print("\n" + "=" * 80)
print("### PART 3: XAC Pro - ML Knowledge Base")
print("=" * 80)
print()

from xac_pro.core.knowledge_base import MLKnowledgeBase

kb = MLKnowledgeBase()

print("📊 Knowledge Base Statistics:")
print(f"  - Concepts: {len(kb.concepts.get('concepts', {}))}")
print(f"  - Algorithms: {len(kb.algorithms.get('algorithms', {}))}")
print(f"  - Metrics: {len(kb.metrics.get('metrics', {}))}")
print()

# Get recommendations
print("🎯 Smart Recommendations:")
print("-" * 80)

algos = kb.get_algorithm_for_problem(
    "classification",
    ["small_data_linear", "high_interpretability"]
)
print(f"Best algorithms for small, interpretable classification: {algos}")

metrics = kb.get_metrics_for_problem("classification", "imbalanced_classes")
print(f"Best metrics for imbalanced classification: {metrics}")
print()

# Search
print("🔍 Search Example: 'forest'")
print("-" * 80)
results = kb.search("forest")
for result in results:
    print(f"  - {result['type']}: {result['name']}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ XAC PRO CAPABILITIES SUMMARY")
print("=" * 80)
print()

print("XAC Classic (v1.0):")
print("  ✓ Explain predictions using SHAP values")
print("  ✓ Evidence-based explanations")
print("  ✓ Safety guardrails")
print()

print("XAC Pro (v2.0) - NEW:")
print("  ✓ Universal ML Teacher (answers ANY question)")
print("  ✓ ML Knowledge Base (15 concepts, 12 algorithms, 12 metrics)")
print("  ✓ Multi-level explanations (beginner/expert/business)")
print("  ✓ Smart troubleshooting")
print("  ✓ Algorithm & metric recommendations")
print("  ✓ 100% local, no API keys needed")
print()

print("Coming Soon:")
print("  ⏳ Auto-discover models from notebooks")
print("  ⏳ CLI interface (xac-pro analyze)")
print("  ⏳ Jupyter magic (%%xac)")
print("  ⏳ Web dashboard (Streamlit)")
print()

print("=" * 80)
print("🎉 XAC Pro - Making ML Interpretable, One Explanation at a Time")
print("=" * 80)
print()
print("Built by Stephen Muema | musyokas753@gmail.com")
print("GitHub: https://github.com/Kaks753/XAC")
