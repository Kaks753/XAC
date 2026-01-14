"""
Guardrails Module for Explainable Analytics Copilot (XAC)

This module enforces safety boundaries and ethical constraints on copilot responses.
It ensures the copilot never predicts, advises, or makes causal claims beyond
what the evidence supports.

Design Rationale:
-----------------
1. Safety first: Multiple layers of protection against harmful outputs
2. Explicit refusals: Clear explanations of why requests are declined
3. Scope enforcement: Copilot stays within explanation role
4. Ethical boundaries: No medical, legal, or financial advice
5. Evidence validation: Ensures all claims are grounded

Author: Muema Stephen
Email: musyokas753@gmail.com
LinkedIn: www.linkedin.com/in/stephen-muema-617339359
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class RefusalReason(Enum):
    """Categorizes why a request was refused."""
    PREDICTION_REQUEST = "prediction_request"
    ADVICE_REQUEST = "advice_request"
    CAUSAL_CLAIM = "causal_claim"
    COUNTERFACTUAL = "counterfactual"
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ETHICAL_BOUNDARY = "ethical_boundary"
    HARMFUL_CONTENT = "harmful_content"


class GuardrailViolation(Exception):
    """
    Exception raised when a guardrail is violated.
    
    This allows the system to gracefully handle violations
    and provide appropriate refusal messages.
    """
    def __init__(self, reason: RefusalReason, message: str):
        self.reason = reason
        self.message = message
        super().__init__(self.message)


class Guardrails:
    """
    Enforces safety boundaries and ethical constraints.
    
    The Guardrails class implements multiple layers of protection:
    1. Query validation (before processing)
    2. Response validation (before returning to user)
    3. Evidence sufficiency checks
    4. Harmful content detection
    
    Design Philosophy:
    ------------------
    "Fail safe": When uncertain, refuse rather than risk harm.
    Transparency: Always explain why a request is refused.
    Consistency: Same request should always get same treatment.
    """
    
    def __init__(self):
        """Initialize guardrails with predefined rules and patterns."""
        self.refusal_patterns = self._initialize_refusal_patterns()
        self.harmful_patterns = self._initialize_harmful_patterns()
        self.ethical_domains = ['medical', 'legal', 'financial']
    
    def _initialize_refusal_patterns(self) -> Dict[RefusalReason, List[str]]:
        """
        Initialize regex patterns for each refusal reason.
        
        Returns:
            Dictionary mapping RefusalReason to list of regex patterns
            
        Design Rationale:
        ----------------
        These patterns catch requests that:
        1. Ask for predictions (out of scope)
        2. Request advice or recommendations (ethical boundary)
        3. Imply causation (not supported by correlational models)
        4. Request counterfactuals/what-ifs (speculative)
        """
        return {
            RefusalReason.PREDICTION_REQUEST: [
                r"\b(predict|forecast|will be|going to be|expect)\b.*\b(value|score|outcome|result)\b",
                r"\bwhat (will|would).*\b(happen|be|occur)\b",
                r"\b(future|tomorrow|next week|next month|next year)\b.*\b(prediction|forecast|value)\b",
                r"\b(estimate|guess|project).*\b(future|upcoming|next)\b",
            ],
            
            RefusalReason.ADVICE_REQUEST: [
                r"\b(should|ought|recommend|advise|suggest|tell me what)\b",
                r"\bwhat (should|ought|must).*\b(do|try|change|improve|fix)\b",
                r"\b(best|optimal|recommended|advised)\b.*\b(action|approach|strategy|decision)\b",
                r"\b(help me decide|which option|what to choose)\b",
            ],
            
            RefusalReason.CAUSAL_CLAIM: [
                r"\b(cause|causes|caused by|because of|leads to|results in)\b",
                r"\b(if.*then|assuming.*will|given.*expect)\b",
                r"\b(effect of|impact of|consequence of).*\bon\b",
                r"\bwhy (did|does|would).*\b(cause|lead to|result in)\b",
            ],
            
            RefusalReason.COUNTERFACTUAL: [
                r"\bwhat if\b",
                r"\b(suppose|imagine|assume|hypothetically)\b",
                r"\b(would have|could have|might have).*\bif\b",
                r"\b(alternative|different scenario|other case)\b.*\bwhat would\b",
            ],
            
            RefusalReason.ETHICAL_BOUNDARY: [
                r"\b(medical|health|diagnosis|treatment|therapy|medication)\b.*\b(advice|recommendation|guidance)\b",
                r"\b(legal|law|lawsuit|litigation|contract)\b.*\b(advice|recommendation|guidance)\b",
                r"\b(financial|investment|stock|trading|portfolio)\b.*\b(advice|recommendation|guidance)\b",
                r"\b(hire|fire|promote|demote|terminate)\b.*\b(should|recommend)\b",
            ],
        }
    
    def _initialize_harmful_patterns(self) -> List[str]:
        """
        Initialize patterns for detecting harmful content.
        
        Returns:
            List of regex patterns for harmful content
            
        Design Rationale:
        ----------------
        Detects content that could:
        1. Discriminate based on protected characteristics
        2. Enable harmful actions
        3. Manipulate or deceive
        4. Violate privacy
        """
        return [
            r"\b(discriminate|bias|prejudice).*\b(race|gender|age|religion|ethnicity|disability)\b",
            r"\b(manipulate|deceive|trick|fool)\b.*\b(people|users|customers|patients)\b",
            r"\b(personal|private|confidential|sensitive)\b.*\b(information|data|records)\b.*\b(access|reveal|expose)\b",
        ]
    
    def validate_query(self, query: str, intent: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a user query before processing.
        
        Args:
            query: User's question
            intent: Classified intent type
            
        Returns:
            Tuple of (is_valid: bool, refusal_message: Optional[str])
            
        Design Choice:
        --------------
        Validation happens BEFORE processing to save computation and
        prevent the system from even attempting to answer inappropriate questions.
        
        Example:
            >>> guardrails = Guardrails()
            >>> valid, msg = guardrails.validate_query("What should I do?", "advice_request")
            >>> print(valid)
            False
        """
        query_lower = query.lower().strip()
        
        # Check if intent is already marked as unsupported
        if intent == "unsupported_request":
            return False, self._generate_refusal_message(
                RefusalReason.OUT_OF_SCOPE,
                query
            )
        
        # Check for harmful content first (highest priority)
        if self._contains_harmful_content(query_lower):
            return False, self._generate_refusal_message(
                RefusalReason.HARMFUL_CONTENT,
                query
            )
        
        # Check each refusal pattern
        for reason, patterns in self.refusal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return False, self._generate_refusal_message(reason, query)
        
        # Query passed all checks
        return True, None
    
    def validate_response(
        self, 
        response: str, 
        evidence: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a response before returning to user.
        
        Args:
            response: Generated response text
            evidence: Evidence dictionary that grounds the response
            
        Returns:
            Tuple of (is_valid: bool, violation_message: Optional[str])
            
        Design Rationale:
        ----------------
        This is a SECOND layer of protection that catches issues in:
        1. Generated responses (if using LLM)
        2. Template-based responses with user data
        3. Ensures no unsupported claims slip through
        """
        response_lower = response.lower()
        
        # Check for prediction language (looking for FUTURE predictions specifically)
        prediction_terms = [
            'will be', 'going to', 'forecast',
            'expect', 'anticipate', 'future value', 'tomorrow', 'next'
        ]
        # Allow "predict" in past tense context (e.g., "predicted" as in "model predicted")
        if any(term in response_lower for term in prediction_terms):
            return False, "Response contains future predictive language"
        
        # Check for advice language
        advice_terms = [
            'should do', 'recommend', 'advise', 'suggest',
            'ought to', 'best to', 'must do'
        ]
        if any(term in response_lower for term in advice_terms):
            return False, "Response contains advice language"
        
        # Check for causal claims
        causal_terms = [
            'causes', 'caused by', 'leads to', 'results in',
            'because of', 'due to', 'effect of'
        ]
        if any(term in response_lower for term in causal_terms):
            return False, "Response contains causal claims"
        
        # If evidence provided, check if response stays grounded
        if evidence:
            valid, msg = self._validate_evidence_alignment(response, evidence)
            if not valid:
                return False, msg
        
        return True, None
    
    def check_evidence_sufficiency(self, evidence: Dict, intent: str) -> Tuple[bool, Optional[str]]:
        """
        Check if evidence is sufficient for the requested explanation type.
        
        Args:
            evidence: Evidence dictionary
            intent: Type of explanation requested
            
        Returns:
            Tuple of (is_sufficient: bool, issue_message: Optional[str])
            
        Design Rationale:
        ----------------
        Different intents require different evidence components:
        - user_explanation: needs prediction + SHAP values
        - feature_importance: needs global importance scores
        - trend_comparison: needs historical data
        - model_performance: needs performance metrics
        """
        required_fields = {
            'user_explanation': ['prediction', 'feature_contributions'],
            'feature_importance': ['feature_contributions'],
            'trend_comparison': ['historical_comparison'],
            'model_performance': ['model_performance'],
        }
        
        if intent not in required_fields:
            return False, f"Unknown intent type: {intent}"
        
        # Check required fields present
        missing_fields = []
        for field in required_fields[intent]:
            if field not in evidence or evidence[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            return False, self._generate_insufficient_evidence_message(missing_fields)
        
        # Additional validation based on intent
        if intent == 'user_explanation':
            if not evidence.get('feature_contributions'):
                return False, "No feature contributions available for explanation"
            if len(evidence['feature_contributions']) == 0:
                return False, "Feature contributions list is empty"
        
        return True, None
    
    def _contains_harmful_content(self, text: str) -> bool:
        """
        Check if text contains potentially harmful content.
        
        Args:
            text: Text to check
            
        Returns:
            True if harmful patterns detected
        """
        for pattern in self.harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _validate_evidence_alignment(
        self, 
        response: str, 
        evidence: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that response stays aligned with evidence.
        
        Args:
            response: Generated response
            evidence: Evidence dictionary
            
        Returns:
            Tuple of (is_aligned: bool, issue_message: Optional[str])
            
        Design Choice:
        --------------
        This is a simple heuristic check. In a production system with LLM
        generation, you might use more sophisticated semantic alignment checks.
        """
        # Check if response mentions confidence and limitations
        # (these should always be included)
        if 'confidence' not in response.lower():
            return False, "Response missing confidence statement"
        
        if 'limitation' not in response.lower() and 'note that' not in response.lower():
            return False, "Response missing limitations disclaimer"
        
        return True, None
    
    def _generate_refusal_message(self, reason: RefusalReason, query: str) -> str:
        """
        Generate appropriate refusal message based on reason.
        
        Args:
            reason: Why the request was refused
            query: Original user query
            
        Returns:
            Professional refusal message
            
        Design Philosophy:
        ------------------
        Refusal messages should:
        1. Be respectful and professional
        2. Explain WHY the request is refused
        3. Suggest what the copilot CAN do
        4. Maintain user trust through transparency
        """
        base_message = "I cannot answer this request. "
        
        reason_explanations = {
            RefusalReason.PREDICTION_REQUEST: (
                "This appears to be a request for a prediction or forecast. "
                "I can only explain existing model outputs, not generate new predictions. "
                "\n\nWhat I can do: Explain why a specific prediction was made, "
                "or describe which features are most important in the model."
            ),
            
            RefusalReason.ADVICE_REQUEST: (
                "This appears to be a request for advice or recommendations. "
                "I am designed to explain model outputs, not provide guidance on actions. "
                "\n\nWhat I can do: Explain what the model predicts and why, "
                "but the decision on how to use this information must be yours."
            ),
            
            RefusalReason.CAUSAL_CLAIM: (
                "This question asks about causal relationships. "
                "My explanations are based on correlational patterns in data, "
                "not causal mechanisms. "
                "\n\nWhat I can do: Explain which features are associated with "
                "predictions, but I cannot claim one thing causes another."
            ),
            
            RefusalReason.COUNTERFACTUAL: (
                "This is a 'what if' or counterfactual question. "
                "I can only explain actual model outputs for real data, "
                "not speculate about hypothetical scenarios. "
                "\n\nWhat I can do: Explain existing predictions and "
                "compare them to historical patterns."
            ),
            
            RefusalReason.ETHICAL_BOUNDARY: (
                "This request involves medical, legal, or financial advice. "
                "I am not qualified to provide such guidance, and doing so "
                "could be harmful. "
                "\n\nPlease consult with qualified professionals in these domains."
            ),
            
            RefusalReason.HARMFUL_CONTENT: (
                "This request contains content that could be harmful or "
                "violate ethical boundaries. "
                "\n\nI am designed to provide safe, responsible explanations only."
            ),
            
            RefusalReason.OUT_OF_SCOPE: (
                "This question is outside my scope of capabilities. "
                "\n\nWhat I can do: Explain specific predictions, "
                "describe feature importance, compare to historical trends, "
                "or provide model performance information."
            ),
            
            RefusalReason.INSUFFICIENT_EVIDENCE: (
                "I don't have sufficient evidence to answer this question reliably. "
                "\n\nI can only provide explanations when complete structured evidence is available."
            ),
        }
        
        return base_message + reason_explanations.get(
            reason, 
            "This request cannot be processed."
        )
    
    def _generate_insufficient_evidence_message(self, missing_fields: List[str]) -> str:
        """
        Generate message explaining missing evidence.
        
        Args:
            missing_fields: List of required fields that are missing
            
        Returns:
            Explanation message
        """
        fields_str = ', '.join(missing_fields)
        return (
            f"Insufficient evidence to generate explanation. "
            f"Missing required fields: {fields_str}. "
            f"\n\nEnsure model outputs include all necessary components "
            f"(predictions, feature contributions, etc.)."
        )


# Example usage and testing
if __name__ == "__main__":
    guardrails = Guardrails()
    
    # Test queries
    test_cases = [
        ("Why is user 102's score 78?", "user_explanation", True),
        ("What will tomorrow's score be?", "prediction_request", False),
        ("What should I do to improve?", "advice_request", False),
        ("What caused the low score?", "causal_claim", False),
        ("What if I changed the value?", "counterfactual", False),
        ("What medical treatment do you recommend?", "ethical_boundary", False),
        ("What are the top features?", "feature_importance", True),
    ]
    
    print("=" * 80)
    print("Guardrails Validation Test Results")
    print("=" * 80)
    
    for query, intent, should_pass in test_cases:
        is_valid, message = guardrails.validate_query(query, intent)
        status = "✓ PASS" if is_valid == should_pass else "✗ FAIL"
        
        print(f"\n{status} | Query: {query}")
        print(f"Intent: {intent} | Valid: {is_valid}")
        if not is_valid:
            print(f"Refusal: {message[:100]}...")
        print("-" * 80)
