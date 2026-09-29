import re
import torch
from typing import Dict, Any
from transformers import pipeline

class TwoTierNLPClassifier:

    def __init__(
        self, 
        emotion_model: str = "bhadresh-savani/distilbert-base-uncased-emotion",
        clinical_model: str = "dsuram/distilbert-mentalhealth-classifier"
    ):
        # 1. Device Selection (CUDA -> MPS -> CPU)
        if torch.cuda.is_available():
            self.device = 0
            print("Loading NLP models onto Nvidia CUDA GPU...")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Loading NLP models onto Apple Silicon GPU (MPS)...")
        else:
            self.device = -1
            print("Loading NLP models onto CPU...")

        # 2. Pipelines
        self.emotion_classifier = pipeline(
            "text-classification",
            model=emotion_model,
            top_k=None,
            device=self.device
        )

        self.clinical_classifier = pipeline(
            "text-classification",
            model=clinical_model,
            top_k=None,
            device=self.device
        )

        # 3. Expanded Crisis Regex (Direct & Indirect Ideation)
        self.hard_crisis_patterns = [
            r"\bsuicide\b", r"\bkill myself\b", r"\bend my life\b", 
            r"\bhurt myself\b", r"\bself-harm\b", r"\bdont want to live\b",
            r"\bdon't want to live\b", r"\bwant to die\b", r"\bi want to die\b",
            r"\bbetter off dead\b", r"\bdont want to wake up\b", r"\bcant go on anymore\b"
        ]

        # 4. Explicit Out-of-Domain Keywords
        self.out_of_domain_triggers = [
            "recipe", "how do i make", "ingredients", "code", "python", 
            "script", "debug", "math", "solve", "capital of", "weather"
        ]

        # 5. Core Mental Health Keywords (For Domain Validation)
        self.mental_health_keywords = [
            "feel", "feeling", "sad", "anxious", "depressed", "stress", 
            "overwhelmed", "worry", "panic", "hopeless", "cbt", "therapy", 
            "thoughts", "die", "suicide", "upset", "angry", "lonely", 
            "exhausted", "sleep", "mind", "life", "pain", "struggling"
        ]

    def warmup(self):
        dummy_text = "Warmup query"
        self.emotion_classifier(dummy_text)
        self.clinical_classifier(dummy_text)
        print("Models warmed up and ready in GPU VRAM.")

    def _check_hard_regex_crisis(self, text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.hard_crisis_patterns)

    def _check_out_of_domain(self, text: str) -> bool:
        text_lower = text.lower()
        
        # Explicit triggers check
        if any(trigger in text_lower for trigger in self.out_of_domain_triggers):
            return True
            
        # If query has >4 words but zero mental health or conversational context
        words = text_lower.split()
        if len(words) > 4:
            has_mental_health_context = any(k in text_lower for k in self.mental_health_keywords)
            if not has_mental_health_context:
                return True
                
        return False

    def classify(self, user_input: str) -> Dict[str, Any]:
        cleaned = user_input.strip()

        # 1. Length outlier check (Allow short queries, reject only blank or massive inputs)
        if len(cleaned) == 0 or len(cleaned) > 1500:
            return {
                "action": "REJECT",
                "is_crisis": False,
                "is_out_of_domain": False,
                "reason": "Invalid string length"
            }

        # 2. 0ms Hard Crisis Scan
        if self._check_hard_regex_crisis(cleaned):
            return {
                "is_crisis": True,
                "is_out_of_domain": False,
                "action": "CRISIS_OVERRIDE",
                "primary_emotion": "severe_distress",
                "clinical_condition": "suicidal_ideation",
                "confidence": 1.0,
                "message": (
                    "It sounds like you are going through an overwhelming moment. "
                    "Please know you don't have to carry this alone:\n\n"
                    "• **National Crisis Lifeline:** Call or text 988 (Available 24/7)\n"
                    "• **Crisis Text Line:** Text HOME to 741741\n"
                    "• **Emergency Services:** Call 911 or visit your nearest emergency room."
                )
            }

        # 3. Out-of-Domain Intercept (e.g., Fruit Salad, Coding)
        if self._check_out_of_domain(cleaned):
            return {
                "is_crisis": False,
                "is_out_of_domain": True,
                "action": "OUT_OF_DOMAIN_REJECT",
                "message": (
                    "I am a specialized CBT AI assistant focused on emotional well-being "
                    "and mental health support. I cannot assist with tasks outside this scope. "
                    "If you'd like to share how you are feeling today, I am here to listen."
                )
            }

        # 4. Surface Emotion Pass
        raw_emotions = self.emotion_classifier(cleaned)[0]
        top_emotion = max(raw_emotions, key=lambda x: x["score"])
        emotion_label = top_emotion["label"].lower()

        clinical_check = ["sadness", "fear", "anger"]

        # 5. Conditional Clinical Pass
        if emotion_label in clinical_check:
            raw_clinical = self.clinical_classifier(cleaned)[0]
            top_clinical = max(raw_clinical, key=lambda x: x["score"])
        else:
            top_clinical = {"label": "Normal", "score": 1.0}
            raw_clinical = [{"label": "Normal", "score": 1.0}]

        clinical_label = top_clinical["label"].lower()

        # Check model-level crisis output fallback
        if clinical_label == "suicidal":
            return {
                "is_crisis": True,
                "is_out_of_domain": False,
                "action": "CRISIS_OVERRIDE",
                "primary_emotion": emotion_label,
                "clinical_condition": "suicidal_ideation",
                "confidence": round(top_clinical["score"], 4),
                "message": (
                    "It sounds like you are going through an overwhelming moment. "
                    "Please know you don't have to carry this alone:\n\n"
                    "• **National Crisis Lifeline:** Call or text 988 (Available 24/7)\n"
                    "• **Crisis Text Line:** Text HOME to 741741\n"
                    "• **Emergency Services:** Call 911 or visit your nearest emergency room."
                )
            }

        # 6. Routing logic
        if clinical_label in ["depression", "anxiety", "ptsd"] or (emotion_label in ["sadness", "fear"] and top_emotion["score"] > 0.80):
            action = "HIGH_PRIORITY_CLINICAL_RAG"
        else:
            action = "STANDARD_RAG"

        return {
            "is_crisis": False,
            "is_out_of_domain": False,
            "action": action,
            "surface_emotion": {
                "label": top_emotion["label"],
                "confidence": round(top_emotion["score"], 4)
            },
            "clinical_nuance": {
                "condition": top_clinical["label"],
                "confidence": round(top_clinical["score"], 4)
            },
            "all_emotion_scores": {e["label"]: round(e["score"], 3) for e in raw_emotions},
            "all_clinical_scores": {c["label"]: round(c["score"], 3) for c in raw_clinical}
        }