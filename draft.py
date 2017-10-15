import pandas as pd

labels = pd.read_csv('../Datas/trainLabels.csv')

# print(labels)
# print(type(labels.label[labels['id'] == 99]))
# print(type(labels.label[labels['id'] == 99]).astype(str))
print(labels[labels['id'] == 99].label.to_string(index = False))
