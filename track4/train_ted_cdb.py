
from transformers import BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate
import os

files = os.listdir('data/raw/tokens')
label_dict = {}
sent_pairs = []

for file in files:
    with open(os.path.join('data/raw/tokens', file), 'r', encoding='utf-8') as f:
        text = ''.join(f.readlines())
    with open(os.path.join('data/Ann/tokens', file), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label = line.split('|')[8]
            if len(label.split('.')) == 1:
                label = 'other'
            else:
                label = label.split('.')[1].split('+')[0]
            first_sent_start = line.split('|')[14].split('..')[0]
            first_sent_end = line.split('|')[14].split('..')[-1]
            first_sent = text[int(first_sent_start):int(first_sent_end)].replace(' ', '').replace('\n', '')
            if line.split('|')[20] == '':
                continue
            second_sent_start = line.split('|')[20].split('..')[0]
            second_sent_end = line.split('|')[20].split('..')[-1]
            second_sent = text[int(second_sent_start):int(second_sent_end)].replace(' ', '').replace('\n', '')
            sent_pairs.append((label, first_sent, second_sent))
            if label not in label_dict:
                label_dict[label] = 1
            else:
                label_dict[label] += 1

print(label_dict)
# {'Synchronous': 744, 
# 'other': 2211, 
# 'Asynchronous': 1605, 
# 'Level-of-detail': 1423, 
# 'Conjunction': 1893, 
# 'Concession': 1637, 
# 'Cause': 2820, 
# 'Condition': 532, 
# 'Equivalence': 149, 
# 'Progression': 335, 
# 'Contrast': 197, 
# 'Substitution': 283, 
# 'Negative-condition': 76, 
# 'Instantiation': 489, 
# 'Disjunction': 185, 
# 'Exception': 36, 
# 'Manner': 188, 
# 'Similarity': 107, 
# 'Purpose': 278, 
# 'Background': 3}

label2id = {}
id2label = {}
idx_label = 0
for k, v in label_dict.items():
    label2id[k] = idx_label
    id2label[idx_label] = k
    idx_label += 1


def gen_train():
    total = len(sent_pairs)
    train_data = sent_pairs[:int(0.9 * total)]
    for data in train_data:
        yield {'label': label2id[data[0]], 'text': data[1], 'text_pair': data[2]}


def gen_dev():
    total = len(sent_pairs)
    dev_data = sent_pairs[int(0.9 * total):]
    for data in dev_data:
        yield {'label': label2id[data[0]], 'text': data[1], 'text_pair': data[2]}


def preprocess_function(examples):
    return tokenizer(text=examples["text"], 
                     text_pair=examples['text_pair'],
                     add_special_tokens=True,
                     truncation=True, 
                     max_length=512)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


ds_train = Dataset.from_generator(gen_train)
ds_dev = Dataset.from_generator(gen_dev)

tokenizer = BertTokenizer.from_pretrained("pretrained_bert_models/chinese_roberta_wwm_large_ext/")

tokenized_train = ds_train.map(preprocess_function, batched=True)
tokenized_dev = ds_dev.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

model = BertForSequenceClassification.from_pretrained("pretrained_bert_models/chinese_roberta_wwm_large_ext/",
                                                      num_labels=len(label_dict), 
                                                      id2label=id2label, 
                                                      label2id=label2id)

training_args = TrainingArguments(
    output_dir="saved_model",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()