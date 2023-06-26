from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import torch
import json
import re

tokenizer = BertTokenizer.from_pretrained("saved_model/checkpoint-98964")
model = BertForSequenceClassification.from_pretrained("saved_model/checkpoint-98964")

w = open("coherence_train.json", 'w+')

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

coherence_features = []
with open("data/train.json", 'r') as f:
    data = f.readlines()
    essays = json.loads(data[0])
    for essay in essays:
        num_pos = 0
        num_neg = 0
        prob_0 = 0
        prob_1 = 0
        paras = ''.join(essay['Text'])
        id = essay["Id"]
        sents = cut_sent(paras)
        for i in range(len(sents)-2):
            text = sents[i] + sents[i+1] + sents[i+2]
            print(text)
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            print(F.softmax(logits, dim=-1))
            prob_0 += F.softmax(logits, dim=-1)[0][0]
            prob_1 += F.softmax(logits, dim=-1)[0][1]
            if predicted_class_id == 0:
                num_neg += 1
            elif predicted_class_id == 1:
                num_pos += 1
        coherence_features.append({"Id":id, 
                                   "features": [num_pos/len(sents), num_neg/len(sents), float(prob_0/len(sents)), float(prob_1/len(sents))]})  

w.write(json.dumps(coherence_features))
w.close()
