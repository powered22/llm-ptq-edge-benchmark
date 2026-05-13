import compressed_tensors
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./results/qwen2.5_1.5b_instruct_gptq_w4a16"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
