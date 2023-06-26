from transformers import BertForTokenClassification, BertTokenizer
from tqdm import tqdm
import torch
import json
import re


def cut_sent(para):
    para = re.sub('([。！？；\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


id2label = {0: "O", 1: "I"}

w = open("track2_0530.json", 'w+')

tokenizer = BertTokenizer.from_pretrained("saved_model/checkpoint-170")
model = BertForTokenClassification.from_pretrained("saved_model/checkpoint-170")

topics = []
with open("data/test.json", 'r') as f:
    data = f.readlines()
    essays = json.loads(data[0])
    for essay in tqdm(essays):
        id = essay["Id"]
        paras = essay['Text']
        title = essay['Title']
        paragraph_topic = []
        for para in paras:
            inputs = tokenizer(title, para, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [id2label[t.item()] for t in predictions[0]][len(title)+2:]
            print(predicted_token_class)
            sents = cut_sent(para)
            num = 0
            sent_count = []
            for i in range(len(sents)):
                sent_count.append((num, num+len(sents[i])))
                num += len(sents[i])
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
            paragraph_topic.append(sents[topic_sent_index])
        if paragraph_topic == []:
            topics.append({"ID":id, 
                           "ParagraphTopic": paragraph_topic,
                           "Full-textTopic": ""})  
        else:
            topics.append({"ID":id, 
                           "ParagraphTopic": paragraph_topic,
                           "Full-textTopic": paragraph_topic[-1]})  
        

w.write(json.dumps(topics, ensure_ascii=False))
w.close()
