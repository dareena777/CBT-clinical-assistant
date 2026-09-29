import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableLambda

from layer1 import TwoTierNLPClassifier
from layer2 import RAGLayer2
from layer3 import CBTInferenceEngine

# Initialize engines once globally
screener = TwoTierNLPClassifier()
screener.warmup()

rag = RAGLayer2(db_path="./chroma_db")
cbt_engine = CBTInferenceEngine(adapter_path="./finetuned_cbt_model")


# 1. State Schema
class GraphState(TypedDict):
    query: str
    is_crisis: bool
    is_out_of_domain: bool
    category: Optional[str]
    target_condition: Optional[str]
    context: str
    internal_thought: Optional[str]
    response: str


# 2. Runnable Functional Logic (Composable with Pipe Operator)
def _classify_input(query: str) -> dict:
    return screener.classify(query)

def _retrieve_rag_context(inputs: dict) -> str:
    chunks = rag.retrieve_context(
        query=inputs["query"],
        category=inputs.get("category"),
        target_condition=inputs.get("target_condition"),
        top_k=2
    )
    return "\n\n".join(chunks) if isinstance(chunks, list) else chunks

def _parse_cbt_generation(raw_output: str) -> dict:
    # 1. Extract <thought>...</thought> blocks
    thought_match = re.search(r'<thought>(.*?)</thought>', raw_output, flags=re.DOTALL)
    extracted_thought = thought_match.group(1).strip() if thought_match else ""

    # 2. Strip <thought> tags
    clean_text = re.sub(r'<thought>.*?</thought>', '', raw_output, flags=re.DOTALL)

    # 3. Strip metadata lines (e.g., "1. Technique: ...")
    strategy_pattern = r'(?im)^\s*(?:\d+[\.\)]\s*)?(?:strategy|technique|cbt technique|cognitive distortion)\s*:.*$'
    clean_text = re.sub(strategy_pattern, '', clean_text)

    # 4. Clean consecutive blank lines
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

    return {"thought": extracted_thought, "response": clean_text}


# Construct LCEL chains for sub-node processing using '|'
classifier_chain = RunnableLambda(lambda state: state["query"]) | RunnableLambda(_classify_input)
post_processor_chain = RunnableLambda(lambda raw: _parse_cbt_generation(raw))


# 3. LangGraph Nodes
def nlp_screener_node(state: GraphState) -> GraphState:
    analysis = classifier_chain.invoke(state)
    new_state = dict(state)
    
    is_crisis = analysis.get("is_crisis", False)
    is_out_of_domain = analysis.get("is_out_of_domain", False)
    
    if is_crisis:
        new_state.update({
            "is_crisis": True,
            "is_out_of_domain": False,
            "category": "crisis",
            "target_condition": "suicidal_ideation",
            "internal_thought": "Crisis input detected. Safety override engaged.",
            "response": analysis.get("message", "Crisis resources provided.")
        })
    elif is_out_of_domain:
        new_state.update({
            "is_crisis": False,
            "is_out_of_domain": True,
            "internal_thought": "Out-of-domain query intercepted.",
            "response": analysis.get("message", "I can only assist with mental health and CBT queries.")
        })
    else:
        new_state.update({
            "is_crisis": False,
            "is_out_of_domain": False,
            "category": analysis.get("surface_emotion", {}).get("label", "general").lower(),
            "target_condition": analysis.get("clinical_nuance", {}).get("condition", "general").lower()
        })
    return new_state


def rag_retrieval_node(state: GraphState) -> GraphState:
    new_state = dict(state)
    new_state["context"] = _retrieve_rag_context(state)
    return new_state


def model_inference_node(state: GraphState) -> GraphState:
    new_state = dict(state)
    raw_output = cbt_engine.generate(
        user_query=state["query"],
        retrieved_context=state["context"]
    )

    # Use post-processing chain constructed with pipe operator
    parsed = post_processor_chain.invoke(raw_output)
    
    new_state["internal_thought"] = parsed["thought"]
    new_state["response"] = parsed["response"]
    return new_state


# 4. Conditional Router Edge
def route_screener_output(state: GraphState) -> str:
    """Short-circuits graph to END when crisis or out-of-domain is triggered."""
    if state.get("is_crisis") or state.get("is_out_of_domain"):
        return "end"
    return "rag"


# 5. Graph Assembly
def build_cbt_graph(checkpointer=None):
    workflow = StateGraph(GraphState)

    # Add individual nodes
    workflow.add_node("screener", nlp_screener_node)
    workflow.add_node("rag", rag_retrieval_node)
    workflow.add_node("model", model_inference_node)

    # Entry edge
    workflow.add_edge(START, "screener")

    # Conditional branching: Screener -> END (for crisis/out-of-domain) OR -> RAG -> Model -> END
    workflow.add_conditional_edges(
        "screener",
        route_screener_output,
        {
            "end": END,
            "rag": "rag"
        }
    )

    workflow.add_edge("rag", "model")
    workflow.add_edge("model", END)

    return workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_cbt_graph()