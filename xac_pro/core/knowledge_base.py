"""
ML Knowledge Base Module for XAC Pro

This module provides access to comprehensive ML knowledge including concepts,
algorithms, metrics, and best practices. Acts as the "brain" of XAC Pro.

Author: Stephen Muema
Email: musyokas753@gmail.com
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KnowledgeItem:
    """Structured knowledge item"""
    name: str
    category: str
    beginner_explanation: str
    expert_explanation: str
    business_explanation: str
    metadata: Dict[str, Any]


class MLKnowledgeBase:
    """
    Universal ML knowledge base for XAC Pro.
    
    Provides access to:
    - ML concepts (overfitting, bias-variance, etc.)
    - Algorithms (XGBoost, Random Forest, etc.)
    - Metrics (accuracy, F1, etc.)
    - Best practices
    
    Design Philosophy:
    - Fast local lookups (no API calls)
    - Multi-level explanations (beginner/expert/business)
    - Rich context for teaching
    """
    
    def __init__(self, knowledge_dir: Optional[str] = None):
        """
        Initialize knowledge base.
        
        Args:
            knowledge_dir: Path to knowledge JSON files
                         If None, uses default location
        """
        if knowledge_dir is None:
            # Default to knowledge/ directory relative to this file
            knowledge_dir = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'knowledge'
            )
        
        self.knowledge_dir = Path(knowledge_dir)
        
        # Load knowledge databases
        self.concepts = self._load_json('ml_concepts.json')
        self.algorithms = self._load_json('algorithms_db.json')
        self.metrics = self._load_json('metrics_db.json')
        
        # Build search indices
        self._build_indices()
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON knowledge file"""
        filepath = self.knowledge_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Creating empty knowledge base.")
            return {}
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _build_indices(self):
        """Build search indices for fast lookups"""
        # Concept index
        self.concept_index = {}
        if 'concepts' in self.concepts:
            for concept_id, concept_data in self.concepts['concepts'].items():
                name = concept_data.get('name', '').lower()
                self.concept_index[name] = concept_id
                self.concept_index[concept_id] = concept_id
        
        # Algorithm index
        self.algorithm_index = {}
        if 'algorithms' in self.algorithms:
            for algo_id, algo_data in self.algorithms['algorithms'].items():
                name = algo_data.get('name', '').lower()
                self.algorithm_index[name] = algo_id
                self.algorithm_index[algo_id] = algo_id
        
        # Metric index
        self.metric_index = {}
        if 'metrics' in self.metrics:
            for metric_id, metric_data in self.metrics['metrics'].items():
                name = metric_data.get('name', '').lower()
                self.metric_index[name] = metric_id
                self.metric_index[metric_id] = metric_id
    
    def explain_concept(
        self, 
        concept: str, 
        level: str = "beginner"
    ) -> Optional[str]:
        """
        Explain an ML concept.
        
        Args:
            concept: Concept name (e.g., "overfitting", "bias_variance_tradeoff")
            level: Explanation level ("beginner", "expert", "business")
            
        Returns:
            Explanation string or None if not found
            
        Example:
            >>> kb = MLKnowledgeBase()
            >>> print(kb.explain_concept("overfitting", level="beginner"))
        """
        concept_id = self.concept_index.get(concept.lower())
        if not concept_id:
            return None
        
        concept_data = self.concepts['concepts'].get(concept_id)
        if not concept_data:
            return None
        
        # Get appropriate explanation level
        explanation_key = f"{level}_explanation"
        explanation = concept_data.get(explanation_key)
        
        if not explanation:
            # Fallback to beginner if level not found
            explanation = concept_data.get('beginner_explanation', 'No explanation available')
        
        # Build rich explanation
        result = f"## {concept_data.get('name', concept)}\n\n"
        result += f"{explanation}\n\n"
        
        # Add symptoms if available
        if 'symptoms' in concept_data:
            result += "**Symptoms:**\n"
            for symptom in concept_data['symptoms']:
                result += f"- {symptom}\n"
            result += "\n"
        
        # Add solutions if available
        if 'solutions' in concept_data:
            result += "**Solutions:**\n"
            for solution in concept_data['solutions']:
                result += f"- {solution}\n"
            result += "\n"
        
        # Add formula for experts
        if level == "expert" and 'mathematical_formula' in concept_data:
            result += f"**Formula:** {concept_data['mathematical_formula']}\n\n"
        
        # Add related concepts
        if 'related_concepts' in concept_data:
            result += "**Related:** " + ", ".join(concept_data['related_concepts']) + "\n"
        
        return result
    
    def explain_algorithm(
        self,
        algorithm: str,
        level: str = "beginner"
    ) -> Optional[str]:
        """
        Explain an ML algorithm.
        
        Args:
            algorithm: Algorithm name (e.g., "xgboost", "random_forest")
            level: Explanation level ("beginner", "expert", "business")
            
        Returns:
            Explanation string or None if not found
        """
        algo_id = self.algorithm_index.get(algorithm.lower())
        if not algo_id:
            return None
        
        algo_data = self.algorithms['algorithms'].get(algo_id)
        if not algo_data:
            return None
        
        # Get explanation
        explanation_key = f"{level}_explanation"
        explanation = algo_data.get(explanation_key, algo_data.get('beginner_explanation', ''))
        
        result = f"## {algo_data.get('name', algorithm)}\n\n"
        result += f"**Type:** {algo_data.get('type', 'Unknown')}\n"
        result += f"**Category:** {algo_data.get('category', 'Unknown')}\n\n"
        result += f"{explanation}\n\n"
        
        # Strengths
        if 'strengths' in algo_data:
            result += "**Strengths:**\n"
            for strength in algo_data['strengths']:
                result += f"✓ {strength}\n"
            result += "\n"
        
        # Weaknesses
        if 'weaknesses' in algo_data:
            result += "**Weaknesses:**\n"
            for weakness in algo_data['weaknesses']:
                result += f"✗ {weakness}\n"
            result += "\n"
        
        # Use cases
        if 'use_cases' in algo_data:
            result += "**Use Cases:**\n"
            for use_case in algo_data['use_cases']:
                result += f"• {use_case}\n"
            result += "\n"
        
        # Key hyperparameters
        if 'hyperparameters' in algo_data:
            result += "**Key Hyperparameters:**\n"
            for param, desc in algo_data['hyperparameters'].items():
                result += f"- `{param}`: {desc}\n"
            result += "\n"
        
        # Sklearn class
        if 'sklearn_class' in algo_data:
            result += f"**Sklearn:** `{algo_data['sklearn_class']}`\n"
        
        return result
    
    def explain_metric(
        self,
        metric: str,
        level: str = "beginner"
    ) -> Optional[str]:
        """
        Explain an ML metric.
        
        Args:
            metric: Metric name (e.g., "accuracy", "f1_score")
            level: Explanation level ("beginner", "expert", "business")
            
        Returns:
            Explanation string or None if not found
        """
        metric_id = self.metric_index.get(metric.lower())
        if not metric_id:
            return None
        
        metric_data = self.metrics['metrics'].get(metric_id)
        if not metric_data:
            return None
        
        # Get explanation
        explanation_key = f"{level}_explanation"
        explanation = metric_data.get(explanation_key, metric_data.get('beginner_explanation', ''))
        
        result = f"## {metric_data.get('name', metric)}\n\n"
        
        # Formula
        if 'formula' in metric_data:
            result += f"**Formula:** `{metric_data['formula']}`\n"
        
        # Range and best value
        if 'range' in metric_data:
            result += f"**Range:** {metric_data['range']}"
            if 'best_value' in metric_data:
                result += f" (best: {metric_data['best_value']})"
            result += "\n\n"
        
        result += f"{explanation}\n\n"
        
        # When to use
        if 'when_to_use' in metric_data:
            result += "**When to Use:**\n"
            for case in metric_data['when_to_use']:
                result += f"✓ {case}\n"
            result += "\n"
        
        # When NOT to use
        if 'when_not_to_use' in metric_data:
            result += "**When NOT to Use:**\n"
            for case in metric_data['when_not_to_use']:
                result += f"✗ {case}\n"
            result += "\n"
        
        # Interpretation guide
        if 'interpretation' in metric_data:
            result += "**Interpretation Guide:**\n"
            interp = metric_data['interpretation']
            if isinstance(interp, dict):
                for value, meaning in interp.items():
                    result += f"- {value}: {meaning}\n"
            else:
                result += str(interp) + "\n"
            result += "\n"
        
        # Sklearn function
        if 'sklearn_function' in metric_data:
            result += f"**Sklearn:** `sklearn.metrics.{metric_data['sklearn_function']}`\n"
        
        return result
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Search across all knowledge types.
        
        Args:
            query: Search query
            
        Returns:
            List of matching items with type and ID
        """
        query_lower = query.lower()
        results = []
        
        # Search concepts
        for concept_id, concept_data in self.concepts.get('concepts', {}).items():
            if query_lower in concept_data.get('name', '').lower():
                results.append({
                    'type': 'concept',
                    'id': concept_id,
                    'name': concept_data.get('name'),
                    'category': concept_data.get('category')
                })
        
        # Search algorithms
        for algo_id, algo_data in self.algorithms.get('algorithms', {}).items():
            if query_lower in algo_data.get('name', '').lower():
                results.append({
                    'type': 'algorithm',
                    'id': algo_id,
                    'name': algo_data.get('name'),
                    'category': algo_data.get('category')
                })
        
        # Search metrics
        for metric_id, metric_data in self.metrics.get('metrics', {}).items():
            if query_lower in metric_data.get('name', '').lower():
                results.append({
                    'type': 'metric',
                    'id': metric_id,
                    'name': metric_data.get('name'),
                    'category': metric_data.get('category')
                })
        
        return results
    
    def get_algorithm_for_problem(
        self,
        problem_type: str,
        constraints: Optional[List[str]] = None
    ) -> List[str]:
        """
        Recommend algorithms for a problem type.
        
        Args:
            problem_type: "classification", "regression", "clustering"
            constraints: ["small_data", "interpretable", "fast", etc.]
            
        Returns:
            List of recommended algorithm IDs
        """
        guide = self.algorithms.get('algorithm_selection_guide', {})
        
        if problem_type not in guide:
            return []
        
        recommendations = []
        problem_guide = guide[problem_type]
        
        if constraints:
            for constraint in constraints:
                if constraint in problem_guide:
                    recommendations.extend(problem_guide[constraint])
        else:
            # Return all for problem type
            for algos in problem_guide.values():
                recommendations.extend(algos)
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_metrics_for_problem(
        self,
        problem_type: str,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Recommend metrics for a problem type.
        
        Args:
            problem_type: "classification", "regression"
            context: "imbalanced", "cost_sensitive", etc.
            
        Returns:
            List of recommended metric names
        """
        guide = self.metrics.get('metrics_selection_guide', {})
        
        if problem_type not in guide:
            return []
        
        problem_guide = guide[problem_type]
        
        if context and context in problem_guide:
            return problem_guide[context]
        
        # Return general purpose metrics
        if 'general_purpose' in problem_guide:
            return problem_guide['general_purpose']
        
        return list(problem_guide.values())[0] if problem_guide else []
    
    def list_all(self, knowledge_type: str) -> List[str]:
        """
        List all items of a knowledge type.
        
        Args:
            knowledge_type: "concepts", "algorithms", "metrics"
            
        Returns:
            List of names
        """
        if knowledge_type == "concepts":
            return [
                data.get('name', id) 
                for id, data in self.concepts.get('concepts', {}).items()
            ]
        elif knowledge_type == "algorithms":
            return [
                data.get('name', id)
                for id, data in self.algorithms.get('algorithms', {}).items()
            ]
        elif knowledge_type == "metrics":
            return [
                data.get('name', id)
                for id, data in self.metrics.get('metrics', {}).items()
            ]
        
        return []


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("XAC Pro ML Knowledge Base Demo")
    print("=" * 80)
    
    # Initialize knowledge base
    kb = MLKnowledgeBase()
    
    # Example 1: Explain a concept
    print("\n### Example 1: Explain Overfitting (Beginner Level)")
    print("-" * 80)
    print(kb.explain_concept("overfitting", level="beginner"))
    
    # Example 2: Explain an algorithm
    print("\n### Example 2: Explain XGBoost (Business Level)")
    print("-" * 80)
    print(kb.explain_algorithm("xgboost", level="business"))
    
    # Example 3: Explain a metric
    print("\n### Example 3: Explain F1 Score")
    print("-" * 80)
    print(kb.explain_metric("f1_score", level="beginner"))
    
    # Example 4: Search
    print("\n### Example 4: Search for 'forest'")
    print("-" * 80)
    results = kb.search("forest")
    for result in results:
        print(f"- {result['type']}: {result['name']}")
    
    # Example 5: Get algorithm recommendations
    print("\n### Example 5: Recommend Algorithms for Classification (Small Data, Interpretable)")
    print("-" * 80)
    algos = kb.get_algorithm_for_problem("classification", ["small_data_linear", "high_interpretability"])
    print("Recommended:", algos)
    
    # Example 6: Get metric recommendations
    print("\n### Example 6: Recommend Metrics for Imbalanced Classification")
    print("-" * 80)
    metrics = kb.get_metrics_for_problem("classification", "imbalanced_classes")
    print("Recommended:", metrics)
    
    print("\n" + "=" * 80)
    print("Knowledge Base loaded successfully!")
    print(f"- Concepts: {len(kb.concepts.get('concepts', {}))}")
    print(f"- Algorithms: {len(kb.algorithms.get('algorithms', {}))}")
    print(f"- Metrics: {len(kb.metrics.get('metrics', {}))}")
