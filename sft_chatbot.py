import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json

# define model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left', torch_dtype='auto', device_map='auto')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')

# move model to device if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load JSONL dataset
def load_jsonl(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

data_path = "oasst1-21k-en/oasst1-21k-en.jsonl"  # Path to your JSONL file
raw_data = load_jsonl(data_path)

# Convert raw data into pairs
def preprocess_data(entry):
    conversation = entry.get("conversations", [])
    human_prompt = ""
    gpt_response = ""
    
    for turn in conversation:
        if turn["from"] == "human":
            human_prompt = turn["value"]
        elif turn["from"] == "gpt":
            gpt_response = turn["value"]
    
    # Skip if either part is missing
    if not human_prompt or not gpt_response:
        return None
    
    # Prepare input-output pair
    input_text = f"User: {human_prompt}\nQwen:"
    output_text = f"{input_text} {gpt_response}"
    
    # Tokenize
    tokenized = tokenizer(output_text, max_length=512, padding="max_length", truncation=True)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Use input_ids as labels
    return tokenized

# Apply preprocessing
processed_data = [preprocess_data(entry) for entry in raw_data]
processed_data = [data for data in processed_data if data is not None]  # Remove None entries

# import pdb; pdb.set_trace()
# Create Hugging Face dataset
train_ind = int(len(processed_data)*0.8)
tokenized_dataset = Dataset.from_list(processed_data[0:train_ind])
tokenized_eval_dataset = Dataset.from_list(processed_data[train_ind:-1])


# Define training arguments
training_args = TrainingArguments(
    output_dir="./qwen_sft_oasst1", # Directory to save the model
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=500, # evaluate after 500 steps
    logging_dir="./logs", # Directory for logs
    per_device_train_batch_size=1, # Adjust based on available GPU memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3, # Number of epochs
    save_steps=1000, # save the model after every 1000 steps
    save_total_limit=3, # keep only the latest 3 ckpt
    warmup_steps=500, # Linear learning rate warmup
    weight_decay=0.01,
    learning_rate=5e-5,
    logging_steps=100,
    # fp16=torch.cuda.is_available(), # enable mixed precision training
    fp16=False,
    bf16=False,  # Disable BF16
    dataloader_drop_last=True,
)

# Trainer for SFT
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save("./qwen_sft_oasst1")
tokenizer.save_pretrained("./qwen_sft_oasst1")