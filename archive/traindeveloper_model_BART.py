import torch
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Step 1: Load and preprocess dataset from Excel
def load_data_from_excel(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    # Rename the columns to 'input' and 'target'
    df = df.rename(columns={'YourInputColumnName': 'input', 'YourTargetColumnName': 'target'})
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.2)

# Step 2: Preprocess data
def preprocess_data(examples, tokenizer, max_input_length=1024, max_target_length=200):
    inputs = tokenizer(examples['input'], max_length=max_input_length, truncation=True, padding="max_length")
    targets = tokenizer(examples['target'], max_length=max_target_length, truncation=True, padding="max_length")
    inputs['labels'] = targets['input_ids']
    return inputs

def fine_tune_bart(train_dataset, test_dataset, output_dir="fine_tuned_bart"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    train_dataset = train_dataset.map(lambda x: preprocess_data(x, tokenizer))
    test_dataset = test_dataset.map(lambda x: preprocess_data(x, tokenizer))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        fp16=True  # Enable mixed precision for faster training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# Step 4: Run Fine-Tuning
if __name__ == "__main__":
    data_file = "C:\\Users\\Deepak J Bhat\\Downloads\\paraphrased_output.xlsx"  # Your dataset file
    datasets = load_data_from_excel(data_file)
    fine_tune_bart(datasets['train'], datasets['test'])
