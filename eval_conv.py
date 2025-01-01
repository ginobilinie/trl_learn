from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import deque


# Load the Qwen model and tokenizer
model_name = "rlhf_fine_tuned_model_qwen_sft_oasst1/checkpoint-6348" #"qwen_sft_oasst1/checkpoint-6348" #"Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_response(prompt, model, tokenizer, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """
    Generate a response from the model given a prompt.
    
    Args:
        prompt (str): Input prompt for the conversation.
        model: The loaded Qwen model.
        tokenizer: The tokenizer associated with the model.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for response randomness.
        top_p (float): Nucleus sampling probability for diverse outputs.
    
    Returns:
        str: Model's response to the prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate a response
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def all_history_conversation():
    print("Qwen Conversation Test")
    print("Type 'exit' to quit.\n")
    
    inference_context = ""  # Keeps track of the conversation history
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Append user input to the context
        inference_context += f"User: {user_input}\nQwen:"
        
        # Generate the model's response
        current_response = generate_response(inference_context, model, tokenizer, max_new_tokens=256)
        print(f"Qwen: {current_response}")
        
        # Update the context with the model's response
        inference_context += current_response + "\n"

def fixed_turn_conversation_history():
    print("Qwen Conversation Test")
    print("Type 'exit' to quit.\n")
    
    # Initialize a deque with a fixed size of 3 turns
    message_queue = deque(maxlen=3)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Add the user's input to the message queue
        message_queue.append(f"User: {user_input}")
        
        # Construct the context from the message queue
        context = "\n".join(message_queue) + "\nQwen:"
        
        # Generate the model's response
        response = generate_response(context, model, tokenizer, max_new_tokens=256)
        
        # Extract only the model's response for the current turn
        current_response = response[len(context):].strip()
        print(f"Qwen: {current_response}")
        
        # Add the model's response to the message queue
        message_queue.append(f"Qwen: {current_response}")


if __name__ == "__main__":
    is_all_history_on = False
    if is_all_history_on:
        all_history_conversation()
    else:
        fixed_turn_conversation_history()