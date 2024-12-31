from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# model loading
model_path = "./rlhf_fine_tuned_model_Qwen/Qwen2.5-1.5B/"
# model_path = 'Qwen/Qwen2.5-1.5B'

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# conduct inference
prompt = "can you explain the concept of reinforcement learning?"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=20,
    num_return_sequences=1,
    temperature=0.7,
    top_k=10,
    top_p=0.9,
)

# import pdb; pdb.set_trace()
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"User query: {prompt} \n\n Model response: {response}")