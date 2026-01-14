import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from copilot import ExplainableAnalyticsCopilot, EvidenceBuilder

# Initialize copilot
copilot = ExplainableAnalyticsCopilot()

# Example evidence (replace with your own project outputs later)
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

response = copilot.explain(
    "Why is user 102's productivity score 78?",
    evidence
)

print(response.explanation)
