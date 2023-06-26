from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

tokenizer = BertTokenizer.from_pretrained("saved_model/best-checkpoint-100")
model = BertForSequenceClassification.from_pretrained("saved_model/best-checkpoint-100")

label2id = {
    '共现关系': 0,
    '反转关系': 1,
    '解说关系': 2,
    '主从关系': 3
}
id2label = {v:k for k, v in label2id.items()}

w = open("track3_0526.json", 'w+')

sent_relations = []
with open("data/test.json", 'r') as f:
    data = f.readlines()
    sent_pairs = json.loads(data[0])
    for pair in sent_pairs:
        ID = pair["ID"]
        PID1 = pair["paragraph1"]["PID"]
        sent1 = pair["paragraph1"]["Text"]
        PID2 = pair["paragraph2"]["PID"]
        sent2 = pair["paragraph2"]["Text"]
        input = tokenizer(text=sent1, text_pair=sent2, add_special_tokens=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**input).logits
        predicted_class_id = logits.argmax().item()
        Relation = id2label[predicted_class_id]
        print(Relation)
        sent_relations.append({"ID":str(ID), "PID1": str(PID1), "PID2": str(PID2), "Relation": Relation})  

w.write(json.dumps(sent_relations, ensure_ascii=False))
w.close()
