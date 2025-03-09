from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import os

class total_summarise:
    def __init__(self, chunk_folder="Roles", model="google/flan-t5-large"):
        self.chunk_folder=chunk_folder
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def get_all_data_and_summarise(self):
        files=os.listdir(self.chunk_folder)
        overlap=50
        chunk_size=512
        summarised_text=""
        for file in files:
            text=""
            filepath=os.path.join(self.chunk_folder,file)
            with open(filepath,"r") as file:
                text+=file.read() +"\n"
            '''
            start=0
            tokens=self.tokenizer.encode(text, truncation=False)
            chunks=[]
            while start<=len(tokens):
                chunk= tokens[start:start+chunk_size]
                print(len(chunk))
                chunks.append(self.tokenizer.decode(chunk,skip_special_tokens=True))
                start+=chunk_size-overlap
            print(chunks)
            summarised_data=[self.summarise_data(chunk) for chunk in chunks]
            '''
            summarised_text+=text
            summarised_text+="\n"+"***********************************************************************************"+"\n"
        return summarised_text


        
    
    def generate_summary(self,text, max_length=10):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=100, truncation=True)
        outputs = self.model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    
    def summarise_data(self, text):
        
        #print(len(text))
        return self.generate_summary(text)
    
    
cc=total_summarise()
summary=cc.summarise_data("document bills issue")
print(summary)
'''
with open("summary.txt","w") as file:
    file.write(summary)
'''

        

        


        