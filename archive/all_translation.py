from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory

# To ensure consistency in language detection
DetectorFactory.seed = 0

# Load MarianMT for translation (English to English translation for other languages)
model_name = 'Helsinki-NLP/opus-mt-de-en'  # Replace {src} with detected language
translator = MarianMTModel.from_pretrained(model_name)
tokenizer_translator = MarianTokenizer.from_pretrained(model_name)

def translate_text(text):
    sentences = text.split(".")  # Split text into sentences (you can improve this with NLP libraries)
    translated_text = []

    for sentence in sentences:
        # Detect language of the sentence
        if detect(sentence) != 'en':  # If sentence is not in English
            # Translate non-English sentences to English
            translated = translator.generate(tokenizer_translator.encode(sentence, return_tensors="pt"))
            translated_sentence = tokenizer_translator.decode(translated[0], skip_special_tokens=True)
            translated_text.append(translated_sentence)
        else:
            # Keep English sentences as is
            translated_text.append(sentence.strip())

    # Join the translated sentences back into a full text
    return ". ".join(translated_text)

input_text=""
# Example input with mixed languages
with open("finetuned_ba_t55.txt","r") as file:
    input_text= file.read()

# Translate the text
translated_text = translate_text(input_text)
print("Translated Text:", translated_text)
