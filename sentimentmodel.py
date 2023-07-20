from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch
import re

class SentimentModel():
    def __init__(self, model_name: str = "oliverguhr/german-sentiment-bert"):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'        
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    def predict_sentiment(self, texts: List[str], output_probabilities = False)-> List[str]:
        texts = [self.clean_text(text) for text in texts]
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        # truncation=True limits number of tokens to model's limitations (512)
        encoded = self.tokenizer.batch_encode_plus(texts, add_special_tokens=False, return_tensors="pt")
        input_id_chunks = list(encoded["input_ids"][0].split(510))
        mask_chunks = list(encoded["attention_mask"][0].split(510))
        for i in range(len(input_id_chunks)):
            input_id_chunks[i]=torch.cat([torch.tensor([1]), input_id_chunks[i], torch.tensor([102])])
            mask_chunks[i]= torch.cat([torch.tensor([1]),mask_chunks[i],torch.tensor([1])])
            pad_len = 512 - input_id_chunks[i].shape[0]
            if pad_len > 0:
                # if padding length is more than 0, we must add padding
                input_id_chunks[i] = torch.cat([input_id_chunks[i], torch.Tensor([0] * pad_len)])
                mask_chunks[i] = torch.cat([mask_chunks[i], torch.Tensor([0] * pad_len)])
        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)
        input_dict = {'input_ids': input_ids.long(),'attention_mask': attention_mask.int()}
                    

        #encoded = input_dict.to(self.device)
        with torch.no_grad():
                outputs = self.model(**input_dict)
                logits=outputs.logits.mean(dim=0)
        label_ids = torch.argmax(outputs[0],axis=1)
        #print(label_ids)
        
        predictions = [torch.softmax(outputs[0].mean(dim=0), dim=-1).tolist()] 
        #print(predictions)
        probabilities = []
        for prediction in predictions:
            #print(prediction)
            probabilities += [[[self.model.config.id2label[index], item] for index, item in enumerate(prediction)]]
            #print(probabilities) 
        sentilabel=max(probabilities[0], key=lambda x: x[1])[0]
            
        if output_probabilities==True: 
            return sentilabel, probabilities
        else:
            return sentilabel
    def replace_numbers(self,text: str) -> str:
            return text.replace("0"," null").replace("1"," eins").replace("2"," zwei")\
                .replace("3"," drei").replace("4"," vier").replace("5"," fünf") \
                .replace("6"," sechs").replace("7"," sieben").replace("8"," acht") \
                .replace("9"," neun")         

    def clean_text(self,text: str)-> str:    
            text = text.replace("\n", " ")        
            text = self.clean_http_urls.sub('',text)
            text = self.clean_at_mentions.sub('',text)        
            text = self.replace_numbers(text)                
            text = self.clean_chars.sub('', text) # use only text chars                          
            text = ' '.join(text.split()) # substitute multiple whitespace with single whitespace   
            text = text.strip().lower()
            return text
