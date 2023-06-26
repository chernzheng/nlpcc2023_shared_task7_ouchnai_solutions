from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

tokenizer = BertTokenizer.from_pretrained("saved_model/epoch1-checkpoint-125")
model = BertForSequenceClassification.from_pretrained("saved_model/epoch1-checkpoint-125")

label2id = {
    '并列关系': 0,
    '顺承关系': 1,
    '递进关系': 2,
    '对比关系': 3,
    '让步关系': 4,
    '转折关系': 5,
    '泛化关系': 6,
    '细化关系': 7,
    '客观因果关系': 8,
    '背景关系': 9,
    '特定条件关系': 10,
    '假设条件关系': 11,
    '主观推论关系': 12
}
id2label = {v:k for k, v in label2id.items()}

w = open("track4_0523.json", 'w+')

sent_relations = []
with open("test.json", 'r') as f:
    data = f.readlines()
    sent_pairs = json.loads(data[0])
    for pair in sent_pairs:
        ID = pair["ID"]
        SID1 = pair["sentence1"]["SID"]
        sent1 = pair["sentence1"]["Text"]
        SID2 = pair["sentence2"]["SID"]
        sent2 = pair["sentence2"]["Text"]
        input = tokenizer(text=sent1, text_pair=sent2, add_special_tokens=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**input).logits
        predicted_class_id = logits.argmax().item()
        Relation = id2label[predicted_class_id]
        print(Relation)
        sent_relations.append({"ID":str(ID), "SID1": str(SID1), "SID2": str(SID2), "Relation": Relation})  

w.write(json.dumps(sent_relations, ensure_ascii=False))
w.close()
