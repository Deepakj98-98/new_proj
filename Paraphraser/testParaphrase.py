import pandas as pd
from transformers import pipeline

# Load Excel File
file_path = "C:\\Users\\Deepak J Bhat\\Downloads\\pasted_parahrase.xlsx"  # Replace with your Excel file path
df = pd.read_excel(file_path)

# Load Hugging Face Paraphrasing Pipeline
paraphraser = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Function to Paraphrase Content
def paraphrase_with_bart(text):
    try:
        result = paraphraser(f"Paraphrase: {text}", max_length=2048, num_return_sequences=1, truncation=True)
        print("done")
        return result[0]['generated_text']
        
    except Exception as e:
        print(f"Error: {e}")
        return text  # Return original text if there's an error

# Paraphrase Each Row
column_to_paraphrase = "Cleaned_Body"  # Replace with the column name you want to paraphrase
output_column = "Paraphrased"       # New column to save paraphrased content
print("here")
df[output_column] = df[column_to_paraphrase].apply(paraphrase_with_bart)

# Save the Result to a New Excel File
output_path = "paraphrased_output.xlsx"
df.to_excel(output_path, index=False)
print(f"Paraphrased data saved to {output_path}")