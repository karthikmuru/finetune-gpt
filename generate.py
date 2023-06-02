import argparse
from torch.utils.data import Dataset
import torch.cuda
import torch
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default=" ", help='Initial text for the model')
    parser.add_argument('--model_dir', type=str, required=True, help='Trained model directory')
    parser.add_argument('--gpt_model', type=str, default="gpt2", help='GPT2 model version')
    parser.add_argument('--max_length', type=int, default=200, help='Max length of generated text')

    
    args = parser.parse_args()
    
    return args


def load_fine_tuned_model(model_dir):
    # Load the fine-tuned GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt_text, max_length=300):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    output = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=max_length, 
        top_p=0.92, 
        top_k=0
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text


if __name__ == '__main__':
    args = setup_argparse()

    model, tokenizer = load_fine_tuned_model(args.model_dir)
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_length)

    print(generated_text)

