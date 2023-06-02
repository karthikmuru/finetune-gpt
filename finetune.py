import argparse
from torch.utils.data import Dataset
import torch.cuda
import torch
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--gpt_model', type=str, default="gpt2", help='GPT2 model version')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    
    args = parser.parse_args()
    
    return args


class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, text, block_size):
        self.examples = tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            max_length=block_size,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.examples.items()}
        return item

def fine_tune_gpt2(text, output_dir, model_name="gpt2", epochs=10, batch_size=2, learning_rate=1e-4):
    # Load the pretrained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_text = tokenizer.encode(text)

    dataset = CustomTextDataset(tokenizer, text, block_size=128)  # Adjust block_size as per your requirements


    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=500,
        save_total_limit=2,
        fp16=True 
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_dir)
    
    del model
    del tokenizer
    del trainer

    print("Fine-tuning complete. Model saved to:", output_dir)


def process_dataset(data_dir) :
    d = pd.read_csv(data_dir, encoding='latin1')
    text = []

    for title in d['track_title'].unique():
        song = ""
        for l in d[d['track_title'] == title]['lyric']:
            song += l + ' \n '
        text.append(song + ' <|endoftext|>')

    return text


if __name__ == '__main__':
    args = setup_argparse()

    text = process_dataset(args.data_dir)
    fine_tune_gpt2(text, args.output_dir, args.gpt_model, args.epochs, args.batch_size, args.learning_rate)