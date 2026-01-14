"""
ML Teacher Module for XAC Pro

This module provides teaching capabilities - answering ANY ML question
using the knowledge base and adaptive explanation depth.

Author: Stephen Muema
Email: musyokas753@gmail.com
"""

from typing import Optional, Dict, List, Tuple
from .knowledge_base import MLKnowledgeBase


class MLTeacher:
    """
    Universal ML teacher that can explain any ML concept, algorithm, or metric.
    
    Features:
    - Adaptive explanation depth (beginner/expert/business)
    - Context-aware responses
    - Suggests related concepts
    - Provides examples and analogies
    
    Design Philosophy:
    - Teach, don't just answer
    - Build understanding progressively
    - Connect related concepts
    - Provide actionable insights
    """
    
    def __init__(self, knowledge_base: Optional[MLKnowledgeBase] = None):
        """
        Initialize ML Teacher.
        
        Args:
            knowledge_base: Optional KB instance, creates new if None
        """
        self.kb = knowledge_base or MLKnowledgeBase()
        
        # Query patterns for different question types
        self.question_patterns = self._init_question_patterns()
    
    def _init_question_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for question classification"""
        return {
            'what_is': [
                'what is', 'what are', 'what does', 'define',
                'explain', 'tell me about', 'describe'
            ],
            'when_to_use': [
                'when should', 'when to use', 'when do i', 'should i use'
            ],
            'how_does_work': [
                'how does', 'how do', 'how can', 'how to'
            ],
            'why': [
                'why is', 'why does', 'why would', 'why do'
            ],
            'compare': [
                'difference between', 'vs', 'versus', 'compare',
                'better than', 'worse than'
            ],
            'which': [
                'which algorithm', 'which metric', 'which model',
                'best algorithm', 'best metric'
            ],
            'troubleshoot': [
                'why is my', 'problem with', 'issue with',
                'not working', 'error', 'low accuracy', 'poor performance'
            ]
        }
    
    def ask(
        self,
        question: str,
        level: str = "beginner",
        context: Optional[Dict] = None
    ) -> str:
        """
        Answer any ML question.
        
        Args:
            question: User's question
            level: "beginner", "expert", or "business"
            context: Optional context (model type, problem, etc.)
            
        Returns:
            Comprehensive answer
            
        Examples:
            >>> teacher = MLTeacher()
            >>> print(teacher.ask("What is overfitting?"))
            >>> print(teacher.ask("When should I use XGBoost?"))
            >>> print(teacher.ask("Why is my recall 0%?"))
        """
        question_lower = question.lower().strip()
        
        # Classify question type
        q_type = self._classify_question(question_lower)
        
        # Route to appropriate handler
        if q_type == 'what_is':
            return self._answer_what_is(question_lower, level)
        elif q_type == 'when_to_use':
            return self._answer_when_to_use(question_lower, level)
        elif q_type == 'how_does_work':
            return self._answer_how_does_work(question_lower, level)
        elif q_type == 'why':
            return self._answer_why(question_lower, level, context)
        elif q_type == 'compare':
            return self._answer_compare(question_lower, level)
        elif q_type == 'which':
            return self._answer_which(question_lower, level, context)
        elif q_type == 'troubleshoot':
            return self._answer_troubleshoot(question_lower, level, context)
        else:
            # General search
            return self._general_search(question_lower, level)
    
    def _classify_question(self, question: str) -> str:
        """Classify question type"""
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in question:
                    return q_type
        return 'general'
    
    def _answer_what_is(self, question: str, level: str) -> str:
        """Answer 'what is' questions"""
        # Extract topic
        for keyword in ['what is', 'what are', 'what does', 'define', 'explain', 'tell me about', 'describe']:
            if keyword in question:
                topic = question.split(keyword)[-1].strip(' ?')
                break
        else:
            topic = question
        
        # Search knowledge base
        # Try concepts first
        explanation = self.kb.explain_concept(topic, level)
        if explanation:
            return explanation
        
        # Try algorithms
        explanation = self.kb.explain_algorithm(topic, level)
        if explanation:
            return explanation
        
        # Try metrics
        explanation = self.kb.explain_metric(topic, level)
        if explanation:
            return explanation
        
        # Fuzzy search
        results = self.kb.search(topic)
        if results:
            response = f"I found these related topics:\n\n"
            for i, result in enumerate(results[:5], 1):
                response += f"{i}. **{result['name']}** ({result['type']})\n"
            response += f"\nTry asking about one of these specifically!"
            return response
        
        return f"I don't have information about '{topic}' yet. Try asking about ML concepts, algorithms, or metrics!"
    
    def _answer_when_to_use(self, question: str, level: str) -> str:
        """Answer 'when to use' questions"""
        # Extract algorithm/metric name
        topic = question.split('use')[-1].strip(' ?')
        
        # Get algorithm info
        explanation = self.kb.explain_algorithm(topic, level)
        if explanation:
            return explanation
        
        # Get metric info
        explanation = self.kb.explain_metric(topic, level)
        if explanation:
            return explanation
        
        return f"I don't have specific guidance on when to use '{topic}'. Try asking about a specific algorithm or metric."
    
    def _answer_how_does_work(self, question: str, level: str) -> str:
        """Answer 'how does it work' questions"""
        # Extract topic
        for keyword in ['how does', 'how do', 'how can', 'how to']:
            if keyword in question:
                topic = question.split(keyword)[-1].split('work')[0].strip(' ?')
                break
        else:
            topic = question
        
        # Get explanation (expert level for 'how it works')
        explanation = self.kb.explain_concept(topic, 'expert')
        if explanation:
            return explanation
        
        explanation = self.kb.explain_algorithm(topic, 'expert')
        if explanation:
            return explanation
        
        return f"I don't have technical details about '{topic}' yet."
    
    def _answer_why(self, question: str, level: str, context: Optional[Dict]) -> str:
        """Answer 'why' questions"""
        # Common 'why' questions about metrics
        if 'recall' in question and ('low' in question or '0' in question or 'zero' in question):
            response = """## Why is Recall Low or 0%?

**Common Causes:**

1. **Class Imbalance** - Model predicts majority class only
   - Solution: Use class_weight='balanced' or SMOTE
   - Check: confusion matrix, class distribution

2. **Insufficient Training Data** for minority class
   - Solution: Collect more data or use data augmentation
   - Check: samples per class

3. **Feature Engineering** - Features don't discriminate classes
   - Solution: Create better features, use domain knowledge
   - Check: feature importance, correlations

4. **Model Too Simple** - Can't capture patterns
   - Solution: Try more complex model (e.g., XGBoost)
   - Check: training vs validation performance

5. **Threshold Too High** - Model is too conservative
   - Solution: Lower decision threshold
   - Check: precision-recall curve

**Quick Fix:**
```python
# Try class weighting
model = LogisticRegression(class_weight='balanced')

# Or adjust threshold
predictions = (model.predict_proba(X)[:, 1] > 0.3).astype(int)
```

**Want More Help?** Tell me about your problem (classes, dataset size, model used).
"""
            return response
        
        elif 'accuracy' in question and ('low' in question or 'poor' in question):
            response = """## Why is Accuracy Low?

**Common Causes:**

1. **Underfitting** - Model too simple
   - Solution: Use more complex model (Random Forest, XGBoost)
   - Check: training accuracy also low?

2. **Poor Feature Engineering**
   - Solution: Create better features, remove noise
   - Check: feature importance, correlation analysis

3. **Not Enough Training Data**
   - Solution: Collect more data or use data augmentation
   - Check: learning curve

4. **Poor Hyperparameters**
   - Solution: Hyperparameter tuning (GridSearch, RandomSearch)
   - Check: default vs tuned performance

5. **Data Quality Issues**
   - Solution: Clean data, handle outliers and missing values
   - Check: data distribution, outliers

**But Wait!** High accuracy isn't always the goal:
- If classes imbalanced, accuracy is misleading
- Focus on precision/recall/F1 instead

**Quick Diagnostic:**
```python
# Check if underfitting
print(f"Train accuracy: {train_acc}")
print(f"Test accuracy: {test_acc}")

# If both low → underfitting
# If train high, test low → overfitting
```
"""
            return response
        
        # General why question - try to find topic
        return self._general_search(question, level)
    
    def _answer_compare(self, question: str, level: str) -> str:
        """Answer comparison questions"""
        # Extract topics being compared
        parts = question.replace(' vs ', ' versus ').split(' versus ')
        if len(parts) == 2:
            topic1 = parts[0].split()[-1].strip()
            topic2 = parts[1].split()[0].strip(' ?')
            
            response = f"## Comparing {topic1.title()} vs {topic2.title()}\n\n"
            
            # Get both explanations
            exp1 = (self.kb.explain_algorithm(topic1, level) or 
                   self.kb.explain_metric(topic1, level) or
                   self.kb.explain_concept(topic1, level))
            
            exp2 = (self.kb.explain_algorithm(topic2, level) or
                   self.kb.explain_metric(topic2, level) or
                   self.kb.explain_concept(topic2, level))
            
            if exp1 and exp2:
                response += f"### {topic1.title()}\n{exp1}\n\n"
                response += f"### {topic2.title()}\n{exp2}\n"
                return response
        
        return "Please rephrase as: 'difference between X and Y' or 'X vs Y'"
    
    def _answer_which(self, question: str, level: str, context: Optional[Dict]) -> str:
        """Answer 'which algorithm/metric' questions"""
        # Determine problem type from context or question
        problem_type = None
        constraints = []
        
        if 'classification' in question:
            problem_type = 'classification'
        elif 'regression' in question:
            problem_type = 'regression'
        elif 'clustering' in question:
            problem_type = 'clustering'
        
        # Extract constraints
        if 'small data' in question or 'little data' in question:
            constraints.append('small_data')
        if 'interpretable' in question or 'explainable' in question:
            constraints.append('high_interpretability')
        if 'fast' in question or 'quick' in question:
            constraints.append('real_time')
        if 'accurate' in question or 'best' in question:
            constraints.append('high_accuracy')
        
        # Get recommendations
        if 'algorithm' in question or 'model' in question:
            if not problem_type:
                return "Please specify the problem type (classification, regression, or clustering)."
            
            recommendations = self.kb.get_algorithm_for_problem(problem_type, constraints or None)
            
            response = f"## Recommended {problem_type.title()} Algorithms\n\n"
            if constraints:
                response += f"**Constraints:** {', '.join(constraints)}\n\n"
            
            response += "**Top Recommendations:**\n"
            for algo in recommendations[:5]:
                algo_data = self.kb.algorithms['algorithms'].get(algo, {})
                name = algo_data.get('name', algo)
                desc = algo_data.get('beginner_explanation', '')
                response += f"\n### {name}\n{desc}\n"
            
            return response
        
        elif 'metric' in question:
            if not problem_type:
                problem_type = 'classification'  # Default
            
            context_str = None
            if 'imbalanced' in question:
                context_str = 'imbalanced_classes'
            elif 'cost' in question:
                context_str = 'cost_sensitive'
            
            recommendations = self.kb.get_metrics_for_problem(problem_type, context_str)
            
            response = f"## Recommended {problem_type.title()} Metrics\n\n"
            response += "**Top Recommendations:**\n"
            for metric in recommendations:
                response += f"- {metric}\n"
            
            return response
        
        return "Ask: 'which algorithm for [problem]?' or 'which metric for [problem]?'"
    
    def _answer_troubleshoot(self, question: str, level: str, context: Optional[Dict]) -> str:
        """Answer troubleshooting questions"""
        # Common issues
        if 'overfitting' in question or 'training accuracy high' in question:
            return self.kb.explain_concept('overfitting', level)
        elif 'underfitting' in question:
            return self.kb.explain_concept('underfitting', level)
        elif 'class imbalance' in question or 'imbalanced' in question:
            return self.kb.explain_concept('class_imbalance', level)
        
        # Use why handler for troubleshooting
        return self._answer_why(question, level, context)
    
    def _general_search(self, question: str, level: str) -> str:
        """General knowledge search"""
        # Extract potential topics from question
        words = question.split()
        
        for word in words:
            word_clean = word.strip('?,.')
            
            # Try each knowledge type
            for method in [self.kb.explain_concept, self.kb.explain_algorithm, self.kb.explain_metric]:
                result = method(word_clean, level)
                if result:
                    return result
        
        # If nothing found, do fuzzy search
        results = self.kb.search(question)
        if results:
            response = "I found these related topics:\n\n"
            for i, result in enumerate(results[:5], 1):
                response += f"{i}. **{result['name']}** ({result['type']})\n"
            response += "\nTry asking about one of these!"
            return response
        
        return """I don't have an answer for that yet. 

**I can help with:**
- ML concepts (overfitting, bias-variance, regularization, etc.)
- Algorithms (XGBoost, Random Forest, Logistic Regression, etc.)
- Metrics (accuracy, precision, recall, F1, etc.)
- When to use which algorithm or metric
- Troubleshooting common ML problems

**Try asking:**
- "What is overfitting?"
- "When should I use XGBoost?"
- "Why is my recall low?"
- "Which algorithm for classification?"
- "Difference between precision and recall?"
"""
    
    def teach_concept(
        self,
        concept: str,
        start_level: str = "beginner",
        include_examples: bool = True
    ) -> str:
        """
        Teach a concept progressively (beginner → expert).
        
        Args:
            concept: Concept name
            start_level: Starting level
            include_examples: Include code examples
            
        Returns:
            Progressive explanation
        """
        response = f"# Learning: {concept.title()}\n\n"
        response += "---\n\n"
        
        # Beginner
        response += "## 🎓 Beginner Level\n\n"
        beginner = self.kb.explain_concept(concept, 'beginner')
        if beginner:
            response += beginner + "\n\n"
        
        response += "---\n\n"
        
        # Business
        response += "## 💼 Business Perspective\n\n"
        business = self.kb.explain_concept(concept, 'business')
        if business:
            response += business + "\n\n"
        
        response += "---\n\n"
        
        # Expert
        response += "## 🔬 Expert Level\n\n"
        expert = self.kb.explain_concept(concept, 'expert')
        if expert:
            response += expert + "\n\n"
        
        return response


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("XAC Pro ML Teacher Demo")
    print("=" * 80)
    
    teacher = MLTeacher()
    
    # Example questions
    questions = [
        "What is overfitting?",
        "When should I use XGBoost?",
        "Why is my recall 0%?",
        "Difference between precision and recall?",
        "Which algorithm for small data classification?",
        "How does SHAP work?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n### Example {i}: {question}")
        print("-" * 80)
        answer = teacher.ask(question, level="beginner")
        print(answer)
        print()
