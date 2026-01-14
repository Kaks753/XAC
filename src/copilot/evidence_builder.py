"""
Evidence Builder Module for Explainable Analytics Copilot (XAC)

This module constructs structured, verifiable evidence from ML model outputs.
Evidence is represented as JSON and includes predictions, feature importance,
SHAP values, historical comparisons, confidence metrics, and limitations.

Design Rationale:
-----------------
1. Evidence-first approach: All explanations must be grounded in data
2. Structured JSON: Machine-readable, version-controllable, reproducible
3. Comprehensive metadata: Includes timestamps, model info, data quality
4. Explicit limitations: Transparency about what the evidence can/cannot show
5. Type safety: Strong typing and validation for reliability

Author: Muema Stephen
Email: musyokas753@gmail.com
LinkedIn: www.linkedin.com/in/stephen-muema-617339359
"""

import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np


class ConfidenceLevel(Enum):
    """Standardized confidence levels for model outputs."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Prediction:
    """
    Represents a single model prediction with metadata.
    
    Design Choice:
    --------------
    Dataclass for type safety, automatic __init__, and easy serialization.
    Stores both the prediction value and contextual information.
    """
    entity_id: str  # ID of the entity (user, customer, case, etc.)
    prediction_value: Union[float, int, str]  # The predicted value
    prediction_type: str  # 'regression', 'classification', 'probability'
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_version: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FeatureContribution:
    """
    Represents the contribution of a single feature to a prediction.
    
    SHAP (SHapley Additive exPlanations) values are used as they:
    1. Have solid theoretical foundation (game theory)
    2. Are model-agnostic
    3. Provide local (instance-level) explanations
    4. Sum to the difference from base prediction
    """
    feature_name: str
    contribution_value: float  # SHAP value or similar
    feature_value: Union[float, int, str, None]  # Actual feature value
    rank: Optional[int] = None  # Rank by absolute contribution
    percentage_contribution: Optional[float] = None  # % of total effect
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class HistoricalComparison:
    """
    Compares current prediction to historical baselines and trends.
    
    Design Rationale:
    ----------------
    Context is critical for interpretation. A "high" score means nothing
    without knowing what's typical. This structure provides:
    - Multiple comparison points (recent vs long-term)
    - Statistical measures (mean, std, percentile)
    - Trend direction
    """
    current_value: float
    baseline_value: float  # Overall historical average
    recent_average: Optional[float] = None  # Last N days/weeks
    percentile_rank: Optional[float] = None  # Where this falls in distribution
    std_deviation: Optional[float] = None
    trend_direction: Optional[str] = None  # 'increasing', 'decreasing', 'stable'
    comparison_period: Optional[str] = None  # e.g., 'last_30_days'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ModelPerformance:
    """
    Model performance metrics and quality indicators.
    
    Different metrics for different problem types:
    - Regression: RMSE, MAE, R²
    - Classification: Accuracy, Precision, Recall, F1, AUC
    - Ranking: NDCG, MAP
    """
    model_type: str  # 'regression', 'classification', 'ranking', etc.
    primary_metric: str  # Main performance metric
    primary_metric_value: float
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_interval: Optional[List[float]] = None  # [lower, upper]
    test_set_size: Optional[int] = None
    last_training_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Evidence:
    """
    Complete evidence package for a single explanation.
    
    This is the core data structure that grounds all explanations.
    Every claim the copilot makes must be traceable to fields in this structure.
    
    Design Philosophy:
    ------------------
    - Comprehensive: Includes all information needed for explanation
    - Structured: Follows consistent schema for reproducibility
    - Transparent: Explicitly lists limitations and assumptions
    - Versioned: Tracks model and data versions for audit trail
    """
    prediction: Prediction
    feature_contributions: List[FeatureContribution]
    confidence_level: ConfidenceLevel
    limitations: List[str]
    
    # Optional components (not always available)
    historical_comparison: Optional[HistoricalComparison] = None
    model_performance: Optional[ModelPerformance] = None
    base_value: Optional[float] = None  # SHAP base value (average prediction)
    data_quality_notes: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    
    # Metadata
    evidence_id: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        """
        Convert evidence to dictionary for JSON serialization.
        
        Handles nested dataclasses and enums properly.
        """
        result = {
            'evidence_id': self.evidence_id,
            'generated_at': self.generated_at,
            'prediction': self.prediction.to_dict(),
            'feature_contributions': [fc.to_dict() for fc in self.feature_contributions],
            'confidence_level': self.confidence_level.value,
            'limitations': self.limitations,
            'base_value': self.base_value,
            'data_quality_notes': self.data_quality_notes,
            'assumptions': self.assumptions,
        }
        
        if self.historical_comparison:
            result['historical_comparison'] = self.historical_comparison.to_dict()
        
        if self.model_performance:
            result['model_performance'] = self.model_performance.to_dict()
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize evidence to JSON string.
        
        Args:
            indent: Number of spaces for indentation (default: 2)
            
        Returns:
            Formatted JSON string
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)


class EvidenceBuilder:
    """
    Constructs Evidence objects from ML model outputs.
    
    This class serves as the interface between raw model outputs
    (numpy arrays, dataframes, etc.) and structured Evidence objects.
    
    Design Rationale:
    ----------------
    - Separation of concerns: Building evidence is separate from explaining it
    - Validation: Ensures evidence meets quality standards before use
    - Flexibility: Can adapt to different model types and data formats
    - Reusability: Same builder works across different ML projects
    """
    
    @staticmethod
    def build_user_explanation_evidence(
        entity_id: str,
        prediction_value: float,
        shap_values: Dict[str, float],
        feature_values: Dict[str, Union[float, int, str]],
        base_value: float,
        historical_data: Optional[Dict[str, float]] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> Evidence:
        """
        Build evidence for explaining a specific prediction.
        
        Args:
            entity_id: Identifier for the entity being predicted
            prediction_value: The model's prediction
            shap_values: Dictionary mapping feature names to SHAP values
            feature_values: Dictionary mapping feature names to actual values
            base_value: SHAP base value (average prediction)
            historical_data: Optional dict with keys like 'baseline', 'recent_avg', 'std'
            model_info: Optional dict with model metadata
            
        Returns:
            Evidence object with complete explanation data
            
        Design Choice:
        --------------
        This method focuses on INSTANCE-LEVEL explanations (why this specific
        prediction?). It prioritizes SHAP values for their theoretical soundness
        and interpretability.
        
        Example:
            >>> evidence = EvidenceBuilder.build_user_explanation_evidence(
            ...     entity_id="user_102",
            ...     prediction_value=78.5,
            ...     shap_values={'task_switching': -0.31, 'sleep_hours': -0.22},
            ...     feature_values={'task_switching': 15, 'sleep_hours': 5.5},
            ...     base_value=88.0,
            ... )
        """
        # Build prediction object
        prediction = Prediction(
            entity_id=entity_id,
            prediction_value=prediction_value,
            prediction_type='regression',
            model_version=model_info.get('version') if model_info else None,
        )
        
        # Build feature contributions
        # Sort by absolute SHAP value to rank by importance
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        total_abs_contribution = sum(abs(v) for v in shap_values.values())
        
        feature_contributions = []
        for rank, (feature_name, shap_value) in enumerate(sorted_features, start=1):
            contribution = FeatureContribution(
                feature_name=feature_name,
                contribution_value=shap_value,
                feature_value=feature_values.get(feature_name),
                rank=rank,
                percentage_contribution=(
                    abs(shap_value) / total_abs_contribution * 100 
                    if total_abs_contribution > 0 else 0
                ),
            )
            feature_contributions.append(contribution)
        
        # Build historical comparison if data provided
        historical_comparison = None
        if historical_data:
            historical_comparison = HistoricalComparison(
                current_value=prediction_value,
                baseline_value=historical_data.get('baseline', base_value),
                recent_average=historical_data.get('recent_avg'),
                std_deviation=historical_data.get('std'),
                percentile_rank=historical_data.get('percentile'),
                comparison_period=historical_data.get('period', 'historical'),
            )
        
        # Determine confidence level
        # Design Choice: Use prediction uncertainty or data quality indicators
        confidence = EvidenceBuilder._calculate_confidence(
            prediction_value=prediction_value,
            base_value=base_value,
            std=historical_data.get('std') if historical_data else None,
        )
        
        # Standard limitations for ML explanations
        limitations = [
            "Explanation is based on correlational patterns, not causal relationships",
            "SHAP values show feature importance for this specific instance",
            "Model assumptions may not hold for out-of-distribution data",
            "Feature interactions beyond pairwise may not be fully captured",
        ]
        
        # Add data quality notes if relevant
        data_quality_notes = []
        if historical_data and historical_data.get('data_quality_issues'):
            data_quality_notes = historical_data['data_quality_issues']
        
        # Build complete evidence
        evidence = Evidence(
            prediction=prediction,
            feature_contributions=feature_contributions,
            confidence_level=confidence,
            limitations=limitations,
            historical_comparison=historical_comparison,
            base_value=base_value,
            data_quality_notes=data_quality_notes if data_quality_notes else None,
        )
        
        return evidence
    
    @staticmethod
    def build_feature_importance_evidence(
        feature_importances: Dict[str, float],
        model_info: Dict[str, Any],
        importance_type: str = 'global',
    ) -> Evidence:
        """
        Build evidence for global feature importance explanation.
        
        Args:
            feature_importances: Dictionary mapping features to importance scores
            model_info: Model metadata (type, performance, etc.)
            importance_type: Type of importance ('global', 'permutation', 'gain')
            
        Returns:
            Evidence object for feature importance explanation
            
        Design Rationale:
        ----------------
        Global feature importance answers "What features matter most OVERALL?"
        Different from SHAP which answers "What mattered for THIS prediction?"
        """
        # Create a dummy prediction (not the focus here)
        prediction = Prediction(
            entity_id='global',
            prediction_value='N/A',
            prediction_type='feature_importance',
            model_version=model_info.get('version'),
        )
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        total_importance = sum(abs(v) for v in feature_importances.values())
        
        feature_contributions = []
        for rank, (feature_name, importance) in enumerate(sorted_features, start=1):
            contribution = FeatureContribution(
                feature_name=feature_name,
                contribution_value=importance,
                feature_value=None,  # Not applicable for global importance
                rank=rank,
                percentage_contribution=(
                    abs(importance) / total_importance * 100 
                    if total_importance > 0 else 0
                ),
            )
            feature_contributions.append(contribution)
        
        # Model performance if available
        model_performance = None
        if 'performance' in model_info:
            perf_data = model_info['performance']
            model_performance = ModelPerformance(
                model_type=model_info.get('type', 'unknown'),
                primary_metric=perf_data.get('primary_metric', 'accuracy'),
                primary_metric_value=perf_data.get('primary_value', 0.0),
                secondary_metrics=perf_data.get('secondary_metrics', {}),
                test_set_size=perf_data.get('test_size'),
                last_training_date=perf_data.get('training_date'),
            )
        
        limitations = [
            f"Feature importance calculated using {importance_type} method",
            "Global importance may not reflect individual prediction patterns",
            "Feature interactions and correlations affect interpretation",
            "Importance scores are relative, not absolute measures",
        ]
        
        evidence = Evidence(
            prediction=prediction,
            feature_contributions=feature_contributions,
            confidence_level=ConfidenceLevel.HIGH,
            limitations=limitations,
            model_performance=model_performance,
        )
        
        return evidence
    
    @staticmethod
    def _calculate_confidence(
        prediction_value: float,
        base_value: float,
        std: Optional[float] = None,
    ) -> ConfidenceLevel:
        """
        Calculate confidence level for a prediction.
        
        Args:
            prediction_value: The predicted value
            base_value: Baseline/average prediction
            std: Standard deviation of historical predictions
            
        Returns:
            ConfidenceLevel enum value
            
        Design Rationale:
        ----------------
        Confidence based on:
        1. Distance from baseline (more extreme = less confident)
        2. Historical variance (high variance = less confident)
        3. Conservative by default (prefer lower confidence when uncertain)
        """
        if std is None or std == 0:
            # No variance data, use moderate confidence
            return ConfidenceLevel.MEDIUM
        
        # Calculate Z-score (how many std deviations from baseline)
        z_score = abs(prediction_value - base_value) / std
        
        if z_score < 0.5:
            return ConfidenceLevel.VERY_HIGH
        elif z_score < 1.0:
            return ConfidenceLevel.HIGH
        elif z_score < 2.0:
            return ConfidenceLevel.MEDIUM
        elif z_score < 3.0:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


# Example usage
if __name__ == "__main__":
    # Example: Building evidence for a user productivity prediction
    print("=" * 80)
    print("Evidence Builder Example: User Productivity Prediction")
    print("=" * 80)
    
    evidence = EvidenceBuilder.build_user_explanation_evidence(
        entity_id="user_102",
        prediction_value=78.5,
        shap_values={
            'task_switching': -0.31,
            'sleep_deficit': -0.22,
            'meeting_hours': -0.15,
            'focus_time': 0.08,
            'collaboration_score': 0.05,
        },
        feature_values={
            'task_switching': 15,
            'sleep_deficit': 2.5,
            'meeting_hours': 4.2,
            'focus_time': 3.5,
            'collaboration_score': 7.8,
        },
        base_value=88.0,
        historical_data={
            'baseline': 88.0,
            'recent_avg': 85.2,
            'std': 5.3,
            'percentile': 35.0,
            'period': 'last_30_days',
        },
        model_info={
            'version': 'v2.1.0',
            'type': 'gradient_boosting',
        },
    )
    
    print("\nGenerated Evidence JSON:\n")
    print(evidence.to_json())
    
    print("\n" + "=" * 80)
    print("Top 3 Feature Contributions:")
    print("=" * 80)
    for fc in evidence.feature_contributions[:3]:
        print(f"\n{fc.rank}. {fc.feature_name}")
        print(f"   SHAP Value: {fc.contribution_value:.3f}")
        print(f"   Actual Value: {fc.feature_value}")
        print(f"   Contribution: {fc.percentage_contribution:.1f}%")
