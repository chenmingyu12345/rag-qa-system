from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen1.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

def build_prompt(query, contexts):

    context_text = "\n".join(contexts)

    prompt = f"""
使用以下上下文回答问题。

上下文：
{context_text}

问题：
{query}

回答：
"""

    return prompt

def generate_answer(prompt):

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    return tokenizer.decode(outputs[0])
