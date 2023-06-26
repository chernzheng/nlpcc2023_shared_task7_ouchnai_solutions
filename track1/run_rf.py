import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

X_train, X_test, y_train, y_test = [], [], [], []

with open('repunc_train.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for d in data:
        X_train.append(d['features'])
        y_train.append(d['Logicgrade'])

with open('repunc_val.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for d in data:
        X_test.append(d['features'])
        y_test.append(d['Logicgrade'])

print(y_train)

with open('coherence_train.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for i in range(len(data)):
        X_train[i].extend(data[i]['features'])

with open('coherence_val.json', 'r') as f:
    data = f.readlines()[0]
    data = json.loads(data)
    for i in range(len(data)):
        X_test[i].extend(data[i]['features'])

regressor = RandomForestClassifier(n_estimators=100, random_state=0)

# fit model
regressor.fit(X_train, y_train)
# make predictions
y_pred = regressor.predict(X_test)

print(y_test)
print(y_pred)
print(f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))
