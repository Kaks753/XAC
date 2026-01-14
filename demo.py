# demo.py
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.copilot import ExplainableAnalyticsCopilot, EvidenceBuilder
    print("✅ Successfully imported ExplainableAnalyticsCopilot!")
    
    # Create a simple example
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
    
    # Test query
    query = "Why is user 102's productivity score 78?"
    response = copilot.explain(query, evidence)
    
    print(f"\n📝 Query: {query}")
    print(f"🔍 Intent: {response.intent}")
    print(f"\n📊 Explanation:")
    print("-" * 50)
    print(response.explanation)
    print("-" * 50)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nChecking project structure...")
    
    # Check what's in src
    import os
    print(f"\nCurrent directory: {os.getcwd()}")
    print(f"src directory exists: {os.path.exists('src')}")
    
    if os.path.exists('src'):
        print("Contents of src:")
        for item in os.listdir('src'):
            print(f"  - {item}")
            
    if os.path.exists('src/copilot'):
        print("\nContents of src/copilot:")
        for item in os.listdir('src/copilot'):
            print(f"  - {item}")
            
except Exception as e:
    print(f"❌ Error: {e}")