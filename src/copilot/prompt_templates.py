"""
Prompt Templates Module for Explainable Analytics Copilot (XAC)

This module converts structured evidence (JSON) into natural language explanations.
Templates are designed to be clear, professional, and grounded in evidence.

Design Rationale:
-----------------
1. Template-based approach: Deterministic, controllable, no LLM required
2. Evidence-grounded: Every claim explicitly references evidence
3. Professional tone: Suitable for business and technical audiences
4. Structured format: Consistent organization across explanations
5. Limitations included: Always transparent about constraints

Author: Muema Stephen
Email: musyokas753@gmail.com
LinkedIn: www.linkedin.com/in/stephen-muema-617339359
"""

from typing import Dict, List, Any
from .evidence_builder import Evidence, ConfidenceLevel


class ExplanationTemplate:
    """
    Base class for explanation templates.
    
    Each template type (user explanation, feature importance, etc.)
    inherits from this class and implements generate_explanation().
    
    Design Philosophy:
    ------------------
    Templates convert structured data → human language while:
    1. Maintaining strict evidence grounding
    2. Using consistent formatting
    3. Including confidence and limitations
    4. Being reusable across different domains
    """
    
    def generate_explanation(self, evidence: Evidence) -> str:
        """
        Generate natural language explanation from evidence.
        
        Args:
            evidence: Evidence object containing all necessary data
            
        Returns:
            Formatted explanation string
            
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_explanation()")
    
    def _format_confidence(self, confidence: ConfidenceLevel) -> str:
        """
        Convert confidence enum to human-readable text.
        
        Args:
            confidence: ConfidenceLevel enum value
            
        Returns:
            Human-readable confidence description
        """
        descriptions = {
            ConfidenceLevel.VERY_HIGH: "very high confidence",
            ConfidenceLevel.HIGH: "high confidence",
            ConfidenceLevel.MEDIUM: "moderate confidence",
            ConfidenceLevel.LOW: "low confidence",
            ConfidenceLevel.VERY_LOW: "very low confidence",
        }
        return descriptions.get(confidence, "unknown confidence")
    
    def _format_limitations(self, limitations: List[str]) -> str:
        """
        Format limitations as a bulleted list.
        
        Args:
            limitations: List of limitation strings
            
        Returns:
            Formatted limitations section
        """
        if not limitations:
            return ""
        
        formatted = "\n\n**Important Limitations:**\n"
        for limitation in limitations:
            formatted += f"- {limitation}\n"
        
        return formatted


class UserExplanationTemplate(ExplanationTemplate):
    """
    Template for explaining a specific model output for an individual case.
    
    This answers: "Why did the model produce X for this particular case?"
    
    Design Rationale:
    ----------------
    Structure:
    1. Summary: Output value and comparison to baseline
    2. Top contributors: Most influential features with SHAP values
    3. Context: Historical comparison if available
    4. Confidence: Model confidence in this output
    5. Limitations: What the explanation can/cannot tell us
    """
    
    def generate_explanation(self, evidence: Evidence) -> str:
        """
        Generate user-specific output explanation.
        
        Args:
            evidence: Evidence object with output value, SHAP, historical data
            
        Returns:
            Natural language explanation
        """
        # Extract data from evidence
        entity_id = evidence.prediction.entity_id
        pred_value = evidence.prediction.prediction_value
        base_value = evidence.base_value or "baseline"
        confidence = self._format_confidence(evidence.confidence_level)
        
        # Build explanation sections
        sections = []
        
        # 1. Summary
        summary = self._generate_summary(entity_id, pred_value, base_value, evidence)
        sections.append(summary)
        
        # 2. Feature contributions
        contributions = self._generate_contributions_section(evidence.feature_contributions)
        sections.append(contributions)
        
        # 3. Historical context (if available)
        if evidence.historical_comparison:
            context = self._generate_historical_context(evidence.historical_comparison)
            sections.append(context)
        
        # 4. Confidence statement
        confidence_stmt = f"\n**Confidence Level:** {confidence.capitalize()}"
        sections.append(confidence_stmt)
        
        # 5. Limitations
        limitations = self._format_limitations(evidence.limitations)
        sections.append(limitations)
        
        return "\n".join(sections)
    
    def _generate_summary(
        self, 
        entity_id: str, 
        pred_value: float, 
        base_value: float,
        evidence: Evidence
    ) -> str:
        """Generate opening summary paragraph."""
        # Calculate difference from baseline if numeric
        try:
            diff = pred_value - base_value
            diff_text = f"{abs(diff):.1f} points {'above' if diff > 0 else 'below'} baseline"
        except (TypeError, ValueError):
            diff_text = "compared to baseline"
        
        return (
            f"## Explanation for {entity_id}\n\n"
            f"The model output shows a value of **{pred_value:.1f}** for {entity_id}, "
            f"which is {diff_text} of {base_value:.1f}. "
            f"This result is based on an analysis of {len(evidence.feature_contributions)} features."
        )
    
    def _generate_contributions_section(self, contributions: List) -> str:
        """Generate feature contributions section."""
        section = "\n### Top Contributing Factors\n\n"
        section += "The following features had the strongest influence on this output:\n\n"
        
        # Show top 5 contributors
        top_n = min(5, len(contributions))
        
        for i in range(top_n):
            fc = contributions[i]
            
            # Format direction
            direction = "increased" if fc.contribution_value > 0 else "decreased"
            
            # Format contribution
            contrib_str = f"{abs(fc.contribution_value):.3f}"
            pct_str = f"{fc.percentage_contribution:.1f}%" if fc.percentage_contribution else ""
            
            section += (
                f"{i+1}. **{fc.feature_name}** (value: {fc.feature_value})\n"
                f"   - Effect: {direction} output by {contrib_str}\n"
            )
            if pct_str:
                section += f"   - Contribution: {pct_str} of total effect\n"
            section += "\n"
        
        return section
    
    def _generate_historical_context(self, comparison) -> str:
        """Generate historical comparison section."""
        section = "\n### Historical Context\n\n"
        
        current = comparison.current_value
        baseline = comparison.baseline_value
        
        # Compare to baseline
        diff = current - baseline
        diff_pct = (diff / baseline * 100) if baseline != 0 else 0
        
        section += (
            f"- **Current value:** {current:.1f}\n"
            f"- **Historical baseline:** {baseline:.1f}\n"
            f"- **Difference:** {diff:+.1f} ({diff_pct:+.1f}%)\n"
        )
        
        # Add recent average if available
        if comparison.recent_average:
            section += f"- **Recent average:** {comparison.recent_average:.1f}\n"
        
        # Add percentile if available
        if comparison.percentile_rank:
            section += f"- **Percentile rank:** {comparison.percentile_rank:.0f}th percentile\n"
        
        return section


class FeatureImportanceTemplate(ExplanationTemplate):
    """
    Template for explaining global feature importance.
    
    This answers: "What features are most important in the model overall?"
    
    Design Rationale:
    ----------------
    Different from user explanation (which is local/instance-specific).
    This shows what matters ACROSS ALL predictions.
    """
    
    def generate_explanation(self, evidence: Evidence) -> str:
        """
        Generate global feature importance explanation.
        
        Args:
            evidence: Evidence object with feature importance scores
            
        Returns:
            Natural language explanation
        """
        sections = []
        
        # 1. Summary
        summary = (
            f"## Model Feature Importance\n\n"
            f"The model uses {len(evidence.feature_contributions)} features for its outputs. "
            f"Below are the features ranked by their overall importance across all model outputs."
        )
        sections.append(summary)
        
        # 2. Top features table
        top_features = self._generate_features_table(evidence.feature_contributions)
        sections.append(top_features)
        
        # 3. Model performance (if available)
        if evidence.model_performance:
            perf = self._generate_performance_section(evidence.model_performance)
            sections.append(perf)
        
        # 4. Limitations
        limitations = self._format_limitations(evidence.limitations)
        sections.append(limitations)
        
        return "\n".join(sections)
    
    def _generate_features_table(self, contributions: List) -> str:
        """Generate formatted table of top features."""
        section = "\n### Top Features by Importance\n\n"
        section += "| Rank | Feature Name | Importance Score | % Contribution |\n"
        section += "|------|--------------|------------------|----------------|\n"
        
        # Show top 10
        top_n = min(10, len(contributions))
        
        for i in range(top_n):
            fc = contributions[i]
            section += (
                f"| {i+1} | {fc.feature_name} | "
                f"{fc.contribution_value:.4f} | "
                f"{fc.percentage_contribution:.1f}% |\n"
            )
        
        return section
    
    def _generate_performance_section(self, performance) -> str:
        """Generate model performance section."""
        section = "\n### Model Performance\n\n"
        
        section += (
            f"- **Model type:** {performance.model_type}\n"
            f"- **Primary metric:** {performance.primary_metric} = {performance.primary_metric_value:.4f}\n"
        )
        
        if performance.secondary_metrics:
            section += "\n**Additional metrics:**\n"
            for metric, value in performance.secondary_metrics.items():
                section += f"- {metric}: {value:.4f}\n"
        
        if performance.test_set_size:
            section += f"\n- **Evaluated on:** {performance.test_set_size} samples\n"
        
        return section


class TrendComparisonTemplate(ExplanationTemplate):
    """
    Template for comparing predictions to historical trends.
    
    This answers: "How does this compare to past patterns?"
    """
    
    def generate_explanation(self, evidence: Evidence) -> str:
        """
        Generate trend comparison explanation.
        
        Args:
            evidence: Evidence with historical comparison data
            
        Returns:
            Natural language explanation
        """
        if not evidence.historical_comparison:
            return "Insufficient historical data for trend comparison."
        
        comp = evidence.historical_comparison
        sections = []
        
        # 1. Summary
        summary = self._generate_trend_summary(comp)
        sections.append(summary)
        
        # 2. Detailed comparison
        details = self._generate_comparison_details(comp)
        sections.append(details)
        
        # 3. Trend interpretation
        if comp.trend_direction:
            trend = self._generate_trend_interpretation(comp)
            sections.append(trend)
        
        # 4. Limitations
        limitations = self._format_limitations(evidence.limitations)
        sections.append(limitations)
        
        return "\n".join(sections)
    
    def _generate_trend_summary(self, comp) -> str:
        """Generate trend summary."""
        current = comp.current_value
        baseline = comp.baseline_value
        diff = current - baseline
        
        if abs(diff) < (comp.std_deviation or 1) * 0.5:
            comparison = "similar to"
        elif diff > 0:
            comparison = "above"
        else:
            comparison = "below"
        
        return (
            f"## Trend Comparison\n\n"
            f"The current value of **{current:.1f}** is {comparison} "
            f"the historical baseline of **{baseline:.1f}**."
        )
    
    def _generate_comparison_details(self, comp) -> str:
        """Generate detailed comparison metrics."""
        section = "\n### Detailed Comparison\n\n"
        
        section += f"- **Current value:** {comp.current_value:.1f}\n"
        section += f"- **Historical baseline:** {comp.baseline_value:.1f}\n"
        
        diff = comp.current_value - comp.baseline_value
        section += f"- **Difference:** {diff:+.1f}\n"
        
        if comp.recent_average:
            section += f"- **Recent average ({comp.comparison_period}):** {comp.recent_average:.1f}\n"
        
        if comp.percentile_rank:
            section += f"- **Percentile:** {comp.percentile_rank:.0f}th\n"
        
        if comp.std_deviation:
            z_score = diff / comp.std_deviation
            section += f"- **Standard deviations from baseline:** {z_score:.2f}\n"
        
        return section
    
    def _generate_trend_interpretation(self, comp) -> str:
        """Generate trend direction interpretation."""
        section = "\n### Trend Direction\n\n"
        
        trend_text = {
            'increasing': 'showing an upward trend',
            'decreasing': 'showing a downward trend',
            'stable': 'remaining stable',
        }.get(comp.trend_direction, 'unclear trend')
        
        section += f"Historical analysis indicates the metric is {trend_text}."
        
        return section


class ModelPerformanceTemplate(ExplanationTemplate):
    """
    Template for explaining model performance and confidence.
    
    This answers: "How reliable is the model?"
    """
    
    def generate_explanation(self, evidence: Evidence) -> str:
        """
        Generate model performance explanation.
        
        Args:
            evidence: Evidence with model performance metrics
            
        Returns:
            Natural language explanation
        """
        if not evidence.model_performance:
            return "Model performance information not available."
        
        perf = evidence.model_performance
        sections = []
        
        # 1. Summary
        summary = (
            f"## Model Performance Report\n\n"
            f"This {perf.model_type} model has been evaluated using multiple metrics. "
            f"Below is a summary of its performance characteristics."
        )
        sections.append(summary)
        
        # 2. Primary metrics
        primary = (
            f"\n### Primary Performance Metric\n\n"
            f"- **{perf.primary_metric}:** {perf.primary_metric_value:.4f}\n"
        )
        
        # Add interpretation if possible
        interpretation = self._interpret_metric(perf.primary_metric, perf.primary_metric_value)
        if interpretation:
            primary += f"\n{interpretation}"
        
        sections.append(primary)
        
        # 3. Secondary metrics
        if perf.secondary_metrics:
            secondary = "\n### Additional Metrics\n\n"
            for metric, value in perf.secondary_metrics.items():
                secondary += f"- **{metric}:** {value:.4f}\n"
            sections.append(secondary)
        
        # 4. Evaluation details
        if perf.test_set_size or perf.last_training_date:
            details = "\n### Evaluation Details\n\n"
            if perf.test_set_size:
                details += f"- **Test set size:** {perf.test_set_size} samples\n"
            if perf.last_training_date:
                details += f"- **Last trained:** {perf.last_training_date}\n"
            sections.append(details)
        
        # 5. Limitations
        limitations = self._format_limitations(evidence.limitations)
        sections.append(limitations)
        
        return "\n".join(sections)
    
    def _interpret_metric(self, metric_name: str, value: float) -> str:
        """Provide interpretation of metric value."""
        metric_lower = metric_name.lower()
        
        if 'accuracy' in metric_lower or 'r2' in metric_lower or 'r²' in metric_lower:
            if value >= 0.9:
                return "This indicates excellent model performance."
            elif value >= 0.7:
                return "This indicates good model performance."
            elif value >= 0.5:
                return "This indicates moderate model performance."
            else:
                return "This indicates room for improvement in model performance."
        
        return ""


# Factory function to get appropriate template
def get_template(intent: str) -> ExplanationTemplate:
    """
    Get the appropriate explanation template for an intent type.
    
    Args:
        intent: Intent type string ('user_explanation', 'feature_importance', etc.)
        
    Returns:
        ExplanationTemplate instance
        
    Design Choice:
    --------------
    Factory pattern allows easy extension with new template types
    while maintaining clean interface.
    """
    templates = {
        'user_explanation': UserExplanationTemplate(),
        'feature_importance': FeatureImportanceTemplate(),
        'trend_comparison': TrendComparisonTemplate(),
        'model_performance': ModelPerformanceTemplate(),
    }
    
    return templates.get(intent, UserExplanationTemplate())


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Prompt Templates Module")
    print("=" * 80)
    print("\nThis module provides templates for converting structured evidence")
    print("into natural language explanations.")
    print("\nAvailable templates:")
    print("- UserExplanationTemplate: Explain specific predictions")
    print("- FeatureImportanceTemplate: Explain global feature importance")
    print("- TrendComparisonTemplate: Compare to historical trends")
    print("- ModelPerformanceTemplate: Explain model quality metrics")
