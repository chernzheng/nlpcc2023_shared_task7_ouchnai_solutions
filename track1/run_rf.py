import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, classification_report

X_train, X_val, y_train, y_val = [], [], [], []

with open('output/repunc_train.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for d in data:
        X_train.append(d['features'][:-4])
        y_train.append(d['Logicgrade'])

with open('output/repunc_val.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for d in data:
        X_val.append(d['features'][:-4])
        y_val.append(d['Logicgrade'])
'''
print(y_train)
with open('output/bi_coherence_train_e4.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for i in range(len(data)):
        X_train[i].extend(data[i]['features'])

with open('output/bi_coherence_val_e4.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for i in range(len(data)):
        X_val[i].extend(data[i]['features'])
'''
with open('output/tri_coherence_train.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for i in range(len(data)):
        X_train[i].append(data[i]['features'][0])

with open('output/tri_coherence_val.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for i in range(len(data)):
        X_val[i].append(data[i]['features'][0])

regressor = RandomForestRegressor(n_estimators=30, random_state=0)

# fit model
regressor.fit(X_train, y_train)
# make predictions
y_pred = [round(i) for i in regressor.predict(X_val)]

print(f1_score(y_val, y_pred, average='macro'))
print(classification_report(y_val, y_pred))
print(regressor.feature_importances_)

X_test, y_test, test_id = [], [], []
with open('output/repunc_test.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for d in data:
        X_test.append(d['features'][:-2])
        test_id.append(d["Id"])
        
with open('output/tri_coherence_test.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for i in range(len(data)):
        X_test[i].append(data[i]['features'][0])

y_pred_test = regressor.predict(X_test)

final = []
for i in range(len(y_pred_test)):
    final.append({"ID": str(test_id[i]), "CoherenceGrade": int(y_pred_test[i])})

print(len(final))

w = open("track1_rfr.json", 'w+')
w.write(json.dumps(final))
w.close()
