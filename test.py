# test_nlp_layer1.py
from layer1 import TwoTierNLPClassifier

def run_tests():
    print("=" * 60)
    print("Initializing Two-Tier NLP Classifier...")
    print("=" * 60)
    
    # Initialize classifier (Will auto-detect MPS / CUDA / CPU)
    classifier = TwoTierNLPClassifier()
    
    # Test cases covering different operational paths
    test_cases = [
        # 1. Standard Input (Normal / Positive)
        {
            "category": "Standard Input",
            "text": "I had a great walk in the park today and felt very refreshed."
        },
        # 2. Mild Distress / Surface Emotion
        {
            "category": "Surface Emotion (Stress/Anxiety)",
            "text": "I'm feeling really worried and anxious about my exam results tomorrow."
        },
        # 3. Deep Clinical Distress (Depression / Hopelessness)
        {
            "category": "Clinical Nuance (Depression)",
            "text": "I feel completely exhausted, empty, and hopeless every single day. I can't get out of bed."
        },
        # 4. Hard Safety Trigger (Instant Override)
        {
            "category": "Crisis Trigger",
            "text": "I don't want to live anymore, I am going to end my life tonight."
        },
        # 5. Outlier Input (Too Short)
        {
            "category": "Outlier Input",
            "text": "hi"
        }
    ]

    print("\nStarting Inferences:\n")
    
    for idx, test in enumerate(test_cases, 1):
        print(f"--- Test #{idx}: [{test['category']}] ---")
        print(f"Input: \"{test['text']}\"")
        
        result = classifier.classify(test['text'])
        
        if result.get("is_crisis"):
            print("🚨 Status: CRISIS TRIGGERED")
            print(f"Action: {result['action']}")
            print(f"Override Output:\n{result['message']}")
        elif result.get("action") == "REJECT":
            print("⚠️ Status: OUTLIER REJECTED")
            print(f"Reason: {result['reason']}")
        else:
            print("✅ Status: PROCEED")
            print(f"Pipeline Action: {result['action']}")
            print(f"Surface Emotion: {result['surface_emotion']['label']} (Confidence: {result['surface_emotion']['confidence']})")
            print(f"Clinical Nuance: {result['clinical_nuance']['condition']} (Confidence: {result['clinical_nuance']['confidence']})")
        
        print("-" * 60 + "\n")

if __name__ == "__main__":
    
    classifier = TwoTierNLPClassifier()
    classifier.warmup()
    result=run_tests()
    
    print(result)