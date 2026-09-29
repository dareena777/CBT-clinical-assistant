import os

# 1. Environment variables to bypass MPS buffer limits
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# Global Configurations
MAX_SEQ_LENGTH = 2048
# Switch to 3B or 1.5B if 7B still hits RAM limits
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "./dataset.jsonl"
OUTPUT_DIR = "./finetuned_cbt_model"


# 1. Inference Engine Class (Apple Silicon / MPS Compatible)
class CBTInferenceEngine:
    def __init__(self, adapter_path: str = OUTPUT_DIR, max_seq_length: int = MAX_SEQ_LENGTH):
        abs_adapter_path = os.path.abspath(adapter_path)
        print(f"Loading base model and attaching adapters from '{abs_adapter_path}'...")
        
        self.max_seq_length = max_seq_length
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map=self.device
        )
        
        self.model = PeftModel.from_pretrained(base_model, abs_adapter_path)
        self.model.eval()

    def generate(self, user_query: str, retrieved_context: str, max_new_tokens: int = 512) -> str:
        messages = [
        {
            "role": "system",
            "content": (
                "You are a dedicated CBT Clinical Assistant specialized EXCLUSIVELY in mental health, "
                "emotional wellness, and cognitive therapy.\n\n"
                "BOUNDARY RULES:\n"
                "1. If the user asks about general topics unrelated to mental health (e.g., programming, "
                "math, recipes, sports, or general trivia), politely decline and state that you are only "
                "trained to assist with mental health and emotional well-being.\n"
                "2. Do NOT output <thought> tags, numbered strategy plans, or internal metadata in your user response."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{retrieved_context}\n\nUser Query: {user_query}"
        }
    ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                use_cache=True
            )

        input_length = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return decoded.strip()


# 2. Fine-Tuning Execution Function
def train_model():
    print("Starting LoRA Fine-Tuning on Apple Silicon (MPS)...")

    abs_dataset_path = os.path.abspath(DATASET_PATH)
    if not os.path.exists(abs_dataset_path):
        raise FileNotFoundError(
            f"Dataset missing at '{abs_dataset_path}'. Please check DATASET_PATH."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Using 'dtype' instead of deprecated 'torch_dtype'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files={"train": abs_dataset_path})
    
    def format_prompts(batch):
        texts = []
        for messages in batch["messages"]:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(formatted)
        return {"text": texts}

    formatted_dataset = dataset["train"].map(format_prompts, batched=True)

    abs_output_dir = os.path.abspath(OUTPUT_DIR)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=1,
        output_dir=abs_output_dir,
        optim="adamw_torch",
        use_cpu=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    trainer.train()

    # Save adapter config & weights
    trainer.model.save_pretrained(abs_output_dir)
    trainer.save_model(abs_output_dir)
    tokenizer.save_pretrained(abs_output_dir)
    print(f"Fine-tuning complete. Adapters saved to {abs_output_dir}")


if __name__ == "__main__":
    train_model()