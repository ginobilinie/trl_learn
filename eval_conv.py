from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Qwen model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B"
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

if __name__ == "__main__":
    print("Qwen Conversation Test")
    print("Type 'exit' to quit.\n")
    
    context = ""  # Keeps track of the conversation history
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Append user input to the context
        context += f"User: {user_input}\nQwen:"
        
        # Generate the model's response
        response = generate_response(context, model, tokenizer)
        print(response)
        
        # Update the context with the model's response
        context += response + "\n"
