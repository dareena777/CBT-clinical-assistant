[ User Input ]
│
▼
┌─────────────────────────────────────────┐
│ Layer 1: Two-Tier NLP Screener         │ ───► (Crisis / Out-of-Domain Override) ──► [ Instant Response ]
└─────────────────────────────────────────┘
│ (Valid CBT Query)
▼
┌─────────────────────────────────────────┐
│ Layer 2: RAG Context Engine (ChromaDB) │ ───► Retrieves CBT Frameworks & Techniques
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│ Layer 3: Fine-Tuned CBT Inference Engine│ ───► Generates Clinical Reasoning & Dialogue
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│ Layer 4: State Graph & Output Parser    │ ───► Strips Internal Tags & Streams Clean Output
└─────────────────────────────────────────┘
