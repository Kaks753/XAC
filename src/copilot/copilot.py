"""
Main Copilot Module for Explainable Analytics Copilot (XAC)

This is the orchestration layer that brings together all components:
intent classification, evidence building, guardrails, and explanation generation.

Design Rationale:
-----------------
1. Clean API: Simple interface for end users
2. Pipeline architecture: Modular stages that can be tested independently
3. Error handling: Graceful failures with informative messages
4. Logging: Track what happens for debugging and audit
5. Extensible: Easy to add new intent types and evidence sources

Author: Muema Stephen
Email: musyokas753@gmail.com
LinkedIn: www.linkedin.com/in/stephen-muema-617339359
"""

import json
import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import asdict

# Fixed relative imports
from .intent import IntentClassifier, IntentType
from .guardrails import Guardrails, GuardrailViolation, RefusalReason
from .evidence_builder import Evidence, EvidenceBuilder
from .prompt_templates import get_template



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CopilotResponse:
    """
    Structured response from the copilot.
    
    Design Rationale:
    ----------------
    Responses are structured objects rather than just strings to:
    1. Include metadata (intent, confidence, etc.)
    2. Support different output formats (text, JSON, etc.)
    3. Enable downstream processing
    4. Maintain audit trail
    """
    
    def __init__(
        self,
        query: str,
        intent: str,
        explanation: str,
        evidence: Optional[Dict] = None,
        confidence: float = 0.0,
        is_refusal: bool = False,
        refusal_reason: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.query = query
        self.intent = intent
        self.explanation = explanation
        self.evidence = evidence
        self.confidence = confidence
        self.is_refusal = is_refusal
        self.refusal_reason = refusal_reason
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary."""
        return {
            'query': self.query,
            'intent': self.intent,
            'explanation': self.explanation,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'is_refusal': self.is_refusal,
            'refusal_reason': self.refusal_reason,
            'metadata': self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def __str__(self) -> str:
        """String representation (just the explanation)."""
        return self.explanation


class ExplainableAnalyticsCopilot:
    """
    Main copilot class that orchestrates the explanation pipeline.
    
    The copilot follows this pipeline:
    1. Classify user intent
    2. Validate query with guardrails
    3. Retrieve/build evidence
    4. Validate evidence sufficiency
    5. Generate explanation from template
    6. Validate response
    7. Return structured response
    
    Design Philosophy:
    ------------------
    - Evidence-first: All explanations must be grounded
    - Safety-first: Multiple validation layers
    - Transparency: Clear about what it can/cannot do
    - Deterministic: Same query + evidence → same explanation
    - Production-ready: Logging, error handling, monitoring hooks
    """
    
    def __init__(self, evidence_provider: Optional[Any] = None):
        """
        Initialize the copilot.
        
        Args:
            evidence_provider: Optional custom evidence provider
                             If None, evidence must be passed to explain()
                             
        Design Choice:
        --------------
        Evidence provider is optional to support two usage patterns:
        1. Integrated: Copilot fetches evidence automatically
        2. External: Caller provides evidence explicitly
        
        Pattern #2 is simpler for demos and gives caller full control.
        """
        self.intent_classifier = IntentClassifier()
        self.guardrails = Guardrails()
        self.evidence_provider = evidence_provider
        
        logger.info("Explainable Analytics Copilot initialized")
    
    def explain(
        self,
        query: str,
        evidence: Optional[Evidence] = None,
        evidence_dict: Optional[Dict] = None,
    ) -> CopilotResponse:
        """
        Main entry point: explain a query using provided evidence.
        
        Args:
            query: User's natural language question
            evidence: Evidence object (structured)
            evidence_dict: Alternative: evidence as dictionary
            
        Returns:
            CopilotResponse with explanation or refusal
            
        Design Choice:
        --------------
        Accept both Evidence objects and dicts for flexibility.
        Evidence objects are type-safe and preferred, but dicts
        allow easier integration with existing systems.
        
        Example:
            >>> copilot = ExplainableAnalyticsCopilot()
            >>> response = copilot.explain(
            ...     "Why is user 102's score low?",
            ...     evidence=my_evidence_object
            ... )
            >>> print(response.explanation)
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Stage 1: Classify intent
            intent, intent_confidence = self._classify_intent(query)
            logger.info(f"Classified intent: {intent.value} (confidence: {intent_confidence:.2f})")
            
            # Stage 2: Validate query
            is_valid, refusal_msg = self._validate_query(query, intent.value)
            if not is_valid:
                logger.warning(f"Query refused: {refusal_msg[:100]}")
                return self._create_refusal_response(
                    query, 
                    intent.value, 
                    refusal_msg
                )
            
            # Stage 3: Get evidence (from parameter or provider)
            if evidence is None and evidence_dict is not None:
                # Convert dict to Evidence if needed
                # (This is simplified; real implementation might be more robust)
                evidence = evidence_dict
            elif evidence is None and self.evidence_provider:
                evidence = self.evidence_provider.get_evidence(query, intent)
            
            if evidence is None:
                logger.error("No evidence provided and no provider configured")
                return self._create_error_response(
                    query,
                    intent.value,
                    "No evidence available for this query"
                )
            
            # Stage 4: Validate evidence sufficiency
            evidence_dict_for_validation = (
                evidence.to_dict() if isinstance(evidence, Evidence) else evidence
            )
            
            is_sufficient, insufficiency_msg = self.guardrails.check_evidence_sufficiency(
                evidence_dict_for_validation,
                intent.value
            )
            
            if not is_sufficient:
                logger.warning(f"Insufficient evidence: {insufficiency_msg}")
                return self._create_error_response(
                    query,
                    intent.value,
                    insufficiency_msg
                )
            
            # Stage 5: Generate explanation
            explanation = self._generate_explanation(intent.value, evidence)
            
            # Stage 6: Validate response
            response_valid, violation_msg = self.guardrails.validate_response(
                explanation,
                evidence_dict_for_validation
            )
            
            if not response_valid:
                logger.error(f"Response validation failed: {violation_msg}")
                return self._create_error_response(
                    query,
                    intent.value,
                    f"Generated explanation failed validation: {violation_msg}"
                )
            
            # Stage 7: Create successful response
            logger.info("Successfully generated explanation")
            return CopilotResponse(
                query=query,
                intent=intent.value,
                explanation=explanation,
                evidence=evidence_dict_for_validation,
                confidence=intent_confidence,
                is_refusal=False,
                metadata={
                    'pipeline_stage': 'completed',
                    'template_type': intent.value,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._create_error_response(
                query,
                'unknown',
                f"An error occurred: {str(e)}"
            )
    
    def _classify_intent(self, query: str) -> Tuple[IntentType, float]:
        """
        Classify query intent.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (IntentType, confidence)
        """
        return self.intent_classifier.classify(query)
    
    def _validate_query(self, query: str, intent: str) -> Tuple[bool, Optional[str]]:
        """
        Validate query passes guardrails.
        
        Args:
            query: User query
            intent: Classified intent
            
        Returns:
            Tuple of (is_valid, refusal_message)
        """
        return self.guardrails.validate_query(query, intent)
    
    def _generate_explanation(self, intent: str, evidence: Evidence) -> str:
        """
        Generate explanation from evidence using appropriate template.
        
        Args:
            intent: Intent type
            evidence: Evidence object
            
        Returns:
            Natural language explanation
        """
        template = get_template(intent)
        explanation = template.generate_explanation(evidence)
        return explanation
    
    def _create_refusal_response(
        self,
        query: str,
        intent: str,
        message: str
    ) -> CopilotResponse:
        """Create a refusal response."""
        return CopilotResponse(
            query=query,
            intent=intent,
            explanation=message,
            is_refusal=True,
            refusal_reason=message,
        )
    
    def _create_error_response(
        self,
        query: str,
        intent: str,
        error_message: str
    ) -> CopilotResponse:
        """Create an error response."""
        explanation = (
            f"I encountered an issue processing your request:\n\n{error_message}\n\n"
            "Please ensure all required data is available and try again."
        )
        
        return CopilotResponse(
            query=query,
            intent=intent,
            explanation=explanation,
            is_refusal=True,
            refusal_reason=error_message,
        )
    
    def get_supported_intents(self) -> Dict[str, str]:
        """
        Get descriptions of supported intent types.
        
        Returns:
            Dictionary mapping intent names to descriptions
        """
        classifier = IntentClassifier()
        return {
            intent.value: classifier.get_intent_description(intent)
            for intent in IntentType
            if intent != IntentType.UNSUPPORTED_REQUEST
        }


# Convenience function for simple usage
def explain_prediction(
    query: str,
    evidence: Evidence
) -> str:
    """
    Simple convenience function for basic usage.
    
    Args:
        query: User question
        evidence: Evidence object
        
    Returns:
        Explanation string
        
    Example:
        >>> explanation = explain_prediction(
        ...     "Why is this score low?",
        ...     my_evidence
        ... )
    """
    copilot = ExplainableAnalyticsCopilot()
    response = copilot.explain(query, evidence)
    return response.explanation


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Explainable Analytics Copilot (XAC)")
    print("=" * 80)
    print("\nInitializing copilot...")
    
    # Create copilot instance
    copilot = ExplainableAnalyticsCopilot()
    
    print("\nSupported Intent Types:")
    print("-" * 80)
    for intent, description in copilot.get_supported_intents().items():
        print(f"\n{intent}:")
        print(f"  {description}")
    
    print("\n" + "=" * 80)
    print("Example: Testing with a valid query")
    print("=" * 80)
    
    # Create sample evidence
    from evidence_builder import Prediction, FeatureContribution, ConfidenceLevel
    
    sample_evidence = Evidence(
        prediction=Prediction(
            entity_id="user_102",
            prediction_value=78.5,
            prediction_type="regression"
        ),
        feature_contributions=[
            FeatureContribution(
                feature_name="task_switching",
                contribution_value=-0.31,
                feature_value=15,
                rank=1,
                percentage_contribution=40.0
            ),
            FeatureContribution(
                feature_name="sleep_deficit",
                contribution_value=-0.22,
                feature_value=2.5,
                rank=2,
                percentage_contribution=28.5
            ),
        ],
        confidence_level=ConfidenceLevel.HIGH,
        limitations=[
            "Explanation based on correlations, not causation",
            "Model assumptions may not hold for unusual cases"
        ],
        base_value=88.0
    )
    
    # Test query
    query = "Why is user 102's productivity score 78?"
    response = copilot.explain(query, sample_evidence)
    
    print(f"\nQuery: {query}")
    print(f"Intent: {response.intent}")
    print(f"Confidence: {response.confidence:.2f}")
    print("\nExplanation:")
    print("-" * 80)
    print(response.explanation)
    
    print("\n" + "=" * 80)
    print("Example: Testing refusal (prediction request)")
    print("=" * 80)
    
    query2 = "What will user 102's score be tomorrow?"
    response2 = copilot.explain(query2, sample_evidence)
    
    print(f"\nQuery: {query2}")
    print(f"Is Refusal: {response2.is_refusal}")
    print("\nResponse:")
    print("-" * 80)
    print(response2.explanation)
