"""
Explainable Analytics Copilot (XAC) Package

A constrained conversational interface for explaining machine learning model outputs
using structured evidence, SHAP values, and explainability techniques.

Main Components:
----------------
- IntentClassifier: Classifies user queries into intent categories
- EvidenceBuilder: Constructs structured evidence from ML outputs
- Guardrails: Enforces safety boundaries and ethical constraints
- ExplanationTemplates: Converts evidence to natural language
- ExplainableAnalyticsCopilot: Main orchestration class

Author: Muema Stephen
Email: musyokas753@gmail.com
LinkedIn: www.linkedin.com/in/stephen-muema-617339359
"""

from .intent import IntentClassifier, IntentType, classify_query
from .evidence_builder import (
    Evidence,
    EvidenceBuilder,
    Prediction,
    FeatureContribution,
    HistoricalComparison,
    ModelPerformance,
    ConfidenceLevel,
)
from .guardrails import Guardrails, RefusalReason, GuardrailViolation
from .prompt_templates import (
    get_template,
    ExplanationTemplate,
    UserExplanationTemplate,
    FeatureImportanceTemplate,
    TrendComparisonTemplate,
    ModelPerformanceTemplate,
)
from .copilot import (
    ExplainableAnalyticsCopilot,
    CopilotResponse,
    explain_prediction,
)

__version__ = "1.0.0"
__author__ = "Muema Stephen"
__email__ = "musyokas753@gmail.com"

__all__ = [
    # Main classes
    'ExplainableAnalyticsCopilot',
    'CopilotResponse',
    
    # Intent classification
    'IntentClassifier',
    'IntentType',
    'classify_query',
    
    # Evidence building
    'Evidence',
    'EvidenceBuilder',
    'Prediction',
    'FeatureContribution',
    'HistoricalComparison',
    'ModelPerformance',
    'ConfidenceLevel',
    
    # Guardrails
    'Guardrails',
    'RefusalReason',
    'GuardrailViolation',
    
    # Templates
    'get_template',
    'ExplanationTemplate',
    'UserExplanationTemplate',
    'FeatureImportanceTemplate',
    'TrendComparisonTemplate',
    'ModelPerformanceTemplate',
    
    # Convenience functions
    'explain_prediction',
]
