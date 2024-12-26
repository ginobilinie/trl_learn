from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from trl import PPOTrainer, PPOConfig
import torch
import os
from copy import deepcopy
from sentence_transformers import SentenceTransformer


def main():
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Load the Dataset
    dataset = load_dataset("hh-rlhf/helpful-base", split="train")

    # Extract prompts and responses
    prompts, chosen_responses, rejected_responses = [], [], []
    for entry in dataset:
        prompt, chosen_response, rejected_response = extract_anthropic_prompt(entry['chosen'], entry['rejected'])
        prompts.append(prompt)
        chosen_responses.append(chosen_response)
        rejected_responses.append(rejected_response)

    # Step 2: Load the Model and Tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)  # Move model to GPU
    model_ref = deepcopy(model).eval()

    # Ensure the tokenizer is valid
    from transformers import PreTrainedTokenizerBase
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("The tokenizer is not a valid PreTrainedTokenizerBase.")

    # Step 3: Configure PPO
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=32,
        mini_batch_size=8,
        gradient_accumulation_steps=4,
        steps=1000,
        log_with="wandb",
    )

    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

    # Load similarity model
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")  # Ensure to load it on the correct device if needed

    def reward_function(prompt, generated, chosen, rejected):
        chosen_similarity = similarity_metric(generated, chosen)
        rejected_similarity = similarity_metric(generated, rejected)
        prompt_relevance = similarity_metric(prompt, generated)
        reward = chosen_similarity - rejected_similarity + 0.5 * prompt_relevance
        return torch.tensor([reward], device=device)  # Ensure reward is on the correct device

    def similarity_metric(response1, response2):
        embedding1 = similarity_model.encode(response1, convert_to_tensor=True).to(device)  # Move to GPU
        embedding2 = similarity_model.encode(response2, convert_to_tensor=True).to(device)  # Move to GPU
        return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()

    # Step 5: Training Loop
    bs = ppo_config.batch_size
    max_length = 1000
    for step in range(ppo_config.steps):
        # Sample a batch of prompts
        batch_prompts = prompts[step*bs: (step+1)*bs]
        batch_chosen = chosen_responses[step*bs: (step+1)*bs]
        batch_rejected = rejected_responses[step*bs: (step+1)*bs]

        # Tokenize prompts and move to GPU
        encoded_prompts = tokenizer(batch_prompts, return_tensors="pt", padding=True, max_length=max_length).input_ids.to(device)
        import pdb; pdb.set_trace()
        # Generate responses for prompts
        generated_responses_tensor = model.generate(encoded_prompts, max_length=max_length)
        generated_responses = tokenizer.batch_decode(generated_responses_tensor, skip_special_tokens=True)

        # Calculate rewards
        rewards = [
            reward_function(p, g, c, r).item()
            for p, g, c, r in zip(batch_prompts, generated_responses, batch_chosen, batch_rejected)
        ]

        rewards_tensors = [torch.tensor(reward).to(device) for reward in rewards]

        # Run PPO optimization step
        encoded_prompts_list = [encoded_prompts[ii] for ii in range(encoded_prompts.shape[0])]
        generated_responses_tensor_list = [generated_responses_tensor[ii] for ii in range(generated_responses_tensor.shape[0])]
        import pdb; pdb.set_trace()
        print(f"Encoded Prompts Shape: {[tensor.shape for tensor in encoded_prompts_list]}")
        print(f"Generated Responses Shape: {[tensor.shape for tensor in generated_responses_tensor_list]}")
        print(f"Rewards Tensors Shape: {[tensor.shape for tensor in rewards_tensors]}")

        ppo_trainer.step(encoded_prompts_list, generated_responses_tensor_list, rewards_tensors) # still to have to figure out this line
        print(f"Step {step}/{ppo_config.steps} complete.")

    # Save the fine-tuned model
    model.save_pretrained("rlhf-finetuned-model")
    tokenizer.save_pretrained("rlhf-finetuned-model")


def extract_anthropic_prompt(chosen, rejected):
    search_term = "\n\nAssistant:"
    common_prefix = os.path.commonprefix([chosen, rejected])
    search_term_idx = common_prefix.rfind(search_term)
    if search_term_idx == -1:
        raise ValueError(f"Invalid data: Prompt does not contain '{search_term}'")
    prompt = chosen[: search_term_idx + len(search_term)]
    chosen_sample = chosen.split('\n\nAssistant:')[-1]
    rejected_sample = rejected.split('\n\nAssistant:')[-1]
    return prompt, chosen_sample, rejected_sample


if __name__ == "__main__":
    main()
