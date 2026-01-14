"""
Intent Classification Module for Explainable Analytics Copilot (XAC)

This module classifies user queries into predefined intent categories to determine
how the copilot should respond. It uses pattern matching and keyword analysis
to identify the user's objective.

Design Rationale:
-----------------
1. Pattern-based approach: Fast, deterministic, and fully local (no API calls)
2. Priority-based matching: More specific patterns checked first
3. Extensible: Easy to add new intents and patterns
4. Conservative classification: Defaults to 'unsupported_request' when uncertain

Author: Muema Stephen
Email: musyokas753@gmail.com
LinkedIn: www.linkedin.com/in/stephen-muema-617339359
"""

import re
from typing import Dict, List, Tuple
from enum import Enum


class IntentType(Enum):
    """
    Enumeration of supported intent types.
    
    Each intent represents a specific type of question the copilot can answer
    using structured evidence and explainability techniques.
    """
    USER_EXPLANATION = "user_explanation"  # Explain specific prediction
    FEATURE_IMPORTANCE = "feature_importance"  # Most important features
    TREND_COMPARISON = "trend_comparison"  # Compare trends over time
    MODEL_PERFORMANCE = "model_performance"  # Model metrics and quality
    UNSUPPORTED_REQUEST = "unsupported_request"  # Cannot answer


class IntentClassifier:
    """
    Classifies user queries into intent categories using pattern matching.
    
    The classifier uses a rule-based approach with regex patterns to identify
    the user's intent. This approach is chosen for its transparency, speed,
    and ability to work completely offline.
    
    Attributes:
        intent_patterns: Dictionary mapping intent types to pattern lists
        refusal_patterns: Patterns that should always result in refusal
    """
    
    def __init__(self):
        """
        Initialize the intent classifier with predefined patterns.
        
        Design Choice: Patterns are hardcoded rather than learned to ensure
        deterministic, explainable behavior and avoid training overhead.
        """
        self.intent_patterns = self._initialize_patterns()
        self.refusal_patterns = self._initialize_refusal_patterns()
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """
        Initialize regex patterns for each intent type.
        
        Returns:
            Dictionary mapping IntentType to list of regex patterns
            
        Design Rationale:
        ----------------
        - Patterns ordered from most to least specific
        - Case-insensitive matching for user convenience
        - Multiple synonyms captured (e.g., "explain", "why", "how come")
        """
        return {
            IntentType.USER_EXPLANATION: [
                r"\b(why|explain|how come|what caused|reason for)\b.*\b(prediction|score|result|outcome|value)\b",
                r"\b(what|which|why).*\b(factors|features|variables|attributes)\b.*\b(contributed|influenced|affected|drove)\b",
                r"\bexplain\b.*\b(user|customer|patient|case|instance)\b.*\d+",
                r"\b(user|customer|patient|case|instance)\b.*\d+.*\b(explanation|breakdown|analysis)\b",
                r"\b(shap|contribution|impact|effect)\b.*\b(values|analysis|breakdown)\b",
            ],
            
            IntentType.FEATURE_IMPORTANCE: [
                r"\b(most|top|key|main|primary|important|significant)\b.*\b(features|factors|variables|attributes|predictors)\b",
                r"\bwhich\b.*\b(features|factors|variables)\b.*\b(important|significant|matter|count)\b",
                r"\b(feature|variable)\b.*\b(importance|ranking|weights|contribution)\b",
                r"\bwhat.*drives\b.*\b(model|prediction|outcome)\b",
            ],
            
            IntentType.TREND_COMPARISON: [
                r"\b(compare|comparison|versus|vs|difference|change)\b.*\b(over time|historical|past|previous|trend)\b",
                r"\b(how.*changed|trend|pattern|evolution)\b.*\b(time|period|month|week|day|year)\b",
                r"\b(baseline|average|typical|normal).*\b(comparison|vs|versus|compared to)\b",
                r"\bcurrent.*\b(vs|versus|compared to).*\b(past|historical|previous|baseline)\b",
            ],
            
            IntentType.MODEL_PERFORMANCE: [
                r"\b(how|what).*\b(accurate|good|well|reliable|performance|quality)\b.*\b(model|prediction|forecast)\b",
                r"\b(model|algorithm)\b.*\b(performance|metrics|accuracy|precision|recall|f1|rmse|mae|r2)\b",
                r"\b(confidence|reliability|uncertainty|error rate)\b.*\b(model|prediction)\b",
                r"\bwhat.*\b(metrics|statistics|measures|scores)\b",
            ],
        }
    
    def _initialize_refusal_patterns(self) -> List[str]:
        """
        Initialize patterns that should always trigger refusal.
        
        Returns:
            List of regex patterns for unsupported queries
            
        Design Rationale:
        ----------------
        These patterns identify queries that:
        1. Request predictions/advice (outside copilot scope)
        2. Ask for causal claims (not supported by correlational models)
        3. Request actions/decisions (ethical boundary)
        4. Ask "what if" or counterfactual questions (speculative)
        """
        return [
            r"\b(should|recommend|advise|suggest|tell me what to|what to do)\b",
            r"\b(predict|forecast|what will|will.*be|future|next)\b.*\b(score|value|outcome|result)\b",
            r"\b(what if|suppose|imagine|assume|hypothetical)\b",
            r"\b(cause|caused by|leads to|results in|will cause)\b",
            r"\b(decide|decision|choose|pick|select)\b.*\b(for me|which one|best option)\b",
            r"\b(medical|legal|financial)\b.*\b(advice|guidance|recommendation)\b",
        ]
    
    def classify(self, query: str) -> Tuple[IntentType, float]:
        """
        Classify a user query into an intent category.
        
        Args:
            query: User's natural language question
            
        Returns:
            Tuple of (IntentType, confidence_score)
            confidence_score in range [0.0, 1.0]
            
        Design Choice:
        --------------
        1. Check refusal patterns first (safety priority)
        2. Then check supported intents
        3. Return first match with confidence based on pattern specificity
        4. Default to UNSUPPORTED_REQUEST if no matches
        
        Example:
            >>> classifier = IntentClassifier()
            >>> intent, conf = classifier.classify("Why is user 102's score low?")
            >>> print(intent)
            IntentType.USER_EXPLANATION
        """
        query_lower = query.lower().strip()
        
        # Safety check: Refuse unsupported queries first
        if self._matches_refusal(query_lower):
            return IntentType.UNSUPPORTED_REQUEST, 0.95
        
        # Check each intent type for matches
        for intent_type, patterns in self.intent_patterns.items():
            match, confidence = self._match_patterns(query_lower, patterns)
            if match:
                return intent_type, confidence
        
        # No matches found
        return IntentType.UNSUPPORTED_REQUEST, 0.90
    
    def _matches_refusal(self, query: str) -> bool:
        """
        Check if query matches refusal patterns.
        
        Args:
            query: Lowercased user query
            
        Returns:
            True if query should be refused
        """
        for pattern in self.refusal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _match_patterns(self, query: str, patterns: List[str]) -> Tuple[bool, float]:
        """
        Check if query matches any pattern in the list.
        
        Args:
            query: Lowercased user query
            patterns: List of regex patterns to check
            
        Returns:
            Tuple of (matched: bool, confidence: float)
            
        Design Choice:
        --------------
        Confidence calculation:
        - First pattern (most specific): 0.95
        - Middle patterns: 0.85
        - Last pattern (least specific): 0.75
        This reflects pattern specificity ordering
        """
        for idx, pattern in enumerate(patterns):
            if re.search(pattern, query, re.IGNORECASE):
                # Calculate confidence based on pattern position
                # Earlier patterns are more specific, get higher confidence
                confidence = 0.95 - (idx * 0.05)
                confidence = max(confidence, 0.75)  # Floor at 0.75
                return True, confidence
        
        return False, 0.0
    
    def get_intent_description(self, intent: IntentType) -> str:
        """
        Get human-readable description of an intent type.
        
        Args:
            intent: IntentType enum value
            
        Returns:
            String description of what the intent represents
        """
        descriptions = {
            IntentType.USER_EXPLANATION: 
                "Explain why a specific prediction was made for a particular case",
            IntentType.FEATURE_IMPORTANCE: 
                "Identify which features are most important in the model",
            IntentType.TREND_COMPARISON: 
                "Compare current predictions to historical trends or baselines",
            IntentType.MODEL_PERFORMANCE: 
                "Provide model performance metrics and confidence levels",
            IntentType.UNSUPPORTED_REQUEST: 
                "Query cannot be answered with available evidence (prediction, advice, causal claim, etc.)",
        }
        return descriptions.get(intent, "Unknown intent type")


def classify_query(query: str) -> Dict[str, any]:
    """
    Convenience function to classify a query and return structured result.
    
    Args:
        query: User's natural language question
        
    Returns:
        Dictionary containing:
        - intent: Intent type as string
        - confidence: Confidence score (0.0-1.0)
        - description: Human-readable intent description
        
    Example:
        >>> result = classify_query("What are the top 5 most important features?")
        >>> print(result['intent'])
        'feature_importance'
    """
    classifier = IntentClassifier()
    intent, confidence = classifier.classify(query)
    
    return {
        'intent': intent.value,
        'confidence': confidence,
        'description': classifier.get_intent_description(intent),
        'is_supported': intent != IntentType.UNSUPPORTED_REQUEST,
    }


# Example usage and testing
if __name__ == "__main__":
    # Test cases covering all intent types
    test_queries = [
        "Why is user 102's productivity score 78?",
        "What are the most important features in the model?",
        "How does this compare to last month's average?",
        "What is the model's accuracy?",
        "What should I do to improve the score?",  # Should be refused
        "Predict tomorrow's value",  # Should be refused
    ]
    
    classifier = IntentClassifier()
    
    print("=" * 80)
    print("Intent Classification Test Results")
    print("=" * 80)
    
    for query in test_queries:
        result = classify_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Supported: {result['is_supported']}")
        print(f"Description: {result['description']}")
        print("-" * 80)
