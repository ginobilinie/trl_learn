# 0. imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from copy import deepcopy
from peft import LoraConfig, TaskType, get_peft_model
from rlhf_learn_gpu import extract_anthropic_prompt
from sentence_transformers import SentenceTransformer


# # 1. load a pretrained model
# # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "gpt2"
# current_device = Accelerator().local_process_index
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#     attn_implementation="flash_attention_2",
# )
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     r=8,
#     target_modules=["q_proj", "v_proj"],
#     lora_alpha=16,
#     lora_dropout=0,
# )
# model = get_peft_model(model, lora_config)
# model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
# model_ref = deepcopy(model).eval()
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 1: Load the Dataset
dataset = load_dataset("hh-rlhf/helpful-base", split="train")

prompts, chosen_responses, rejected_responses = [], [], []
for entry in dataset:
    prompt, chosen_response, rejected_response = extract_anthropic_prompt(entry['chosen'], entry['rejected'])
    prompts.append(prompt)
    chosen_responses.append(chosen_response)
    rejected_responses.append(rejected_response)
    
# 1. load a pretrained model
model_name = "gpt2"
# current_device = Accelerator().local_process_index
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)  # Move model to GPU
model_ref = deepcopy(model).eval()

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")  # Ensure to load it on the correct device if needed

def reward_function(prompt, generated, chosen, rejected):
    # import pdb; pdb.set_trace()
    chosen_similarity = similarity_metric(generated, chosen)
    rejected_similarity = similarity_metric(generated, rejected)
    prompt_relevance = similarity_metric(prompt, generated)
    reward = chosen_similarity - rejected_similarity + 0.5 * prompt_relevance
    return torch.tensor([reward], device=device)  # Ensure reward is on the correct device

def similarity_metric(response1, response2):
    embedding1 = similarity_model.encode(response1, convert_to_tensor=True).to(device)  # Move to GPU
    embedding2 = similarity_model.encode(response2, convert_to_tensor=True).to(device)  # Move to GPU
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()

# 3. encode a query
for query_txt, chosen_txt, rejected_txt in zip(prompts, chosen_responses, rejected_responses):
    # query_txt = "This morning I went to the "
    query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(
        model.pretrained_model.device
    )

    # 4. generate model response
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 20,
    }
    response_tensor = ppo_trainer.generate(
        [item for item in query_tensor], return_prompt=False, **generation_kwargs
    )
    response_txt = tokenizer.decode(response_tensor[0])

    # 5. define a reward for response
    # (this could be any reward such as human feedback or output from another model)
    # import pdb; pdb.set_trace()
    rewards = [
        reward_function(p, g, c, r).item()
        for p, g, c, r in zip([query_txt], [response_txt], [chosen_txt], [rejected_txt])
    ]
    rewards_tensors = [torch.tensor(reward).to(device) for reward in rewards]
    # reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

    # 6. train model with ppo
    # import pdb; pdb.set_trace()
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], rewards_tensors)

    print(f"train_stats: {train_stats}")

