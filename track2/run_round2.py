from transformers import BertForTokenClassification, BertTokenizer
from tqdm import tqdm
import torch
import json
import re

id2label = {0: "O", 1: "I"}

w = open("track2_0531.json", 'w+')

tokenizer = BertTokenizer.from_pretrained("saved_model/round2-checkpoint-26")
model = BertForTokenClassification.from_pretrained("saved_model/round2-checkpoint-26")

titles = []
with open("data/test.json", 'r') as f:
    data = f.readlines()
    essays = json.loads(data[0])
    for essay in essays:
        title = essay["Title"]
        titles.append(title)

topics = []
with open("output/track2_0530.json", 'r') as f:
    data = f.readlines()
    essays = json.loads(data[0])
    for index, essay in enumerate(essays):
        if essay['ParagraphTopic'] == []:
            topics.append({"ID": essay["ID"], 
                           "ParagraphTopic": [],
                           "Full-textTopic": ""})
            continue
        paragraph_topic_sents = ''.join(essay['ParagraphTopic'])
        title = titles[index]
        inputs = tokenizer(title, paragraph_topic_sents, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [id2label[t.item()] for t in predictions[0]][len(title)+2:]
        print(predicted_token_class)
        
        sents = essay['ParagraphTopic']
        num = 0
        sent_count = []
        for s in sents:
            sent_count.append((num, num+len(s)))
            num += len(s)
        sent_ratios = []
        for i, j in sent_count:
            num_o = 0
            num_i = 0
            # print(predicted_token_class[i:j])
            for label in predicted_token_class[i:j]:
                if label == 'O':
                    num_o += 1
                elif label == 'I':
                    num_i += 1
            # print(num_i, num_o)
            if num_i+num_o == 0:
                sent_ratios.append(0)
            else:
                sent_ratios.append(num_i/(num_i+num_o))
        print(max(sent_ratios))
        topic_sent_index = sent_ratios.index(max(sent_ratios))
        full_text_topic = sents[topic_sent_index]
        topics.append({"ID": essay["ID"], 
                        "ParagraphTopic": essay['ParagraphTopic'],
                        "Full-textTopic": full_text_topic})

w.write(json.dumps(topics, ensure_ascii=False))
w.close()
