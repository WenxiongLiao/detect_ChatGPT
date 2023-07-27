from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd
from collections import Counter
import torch
import matplotlib.pyplot as plt


plt.rc('font',family='Times New Roman')

#定义函数来显示柱状上的数值
def autolabel(rects):
 for rect in rects:
  height = rect.get_height()
  plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

def show(sub_label,x_bar,y_bar,ylabel,title):
    total_width, n = 1, 3
    width = total_width / n
    x= list(range(len(sub_label)))
    plt.rc('font', size=10)
    plt.figure(figsize=(4, 2))
    a=plt.bar(x, x_bar, width=width, label='human',fc = '#aadfff')
    for i in range(len(x)):
        x[i] = x[i] + width 
    b=plt.bar(x, y_bar, width=width, label='ChatGPT',fc = '#ed7857')
    for i in range(len(x)):
        x[i] = x[i] - width * 1 / 2 
    plt.xticks(x, sub_label, rotation=30)
    # autolabel(a)
    # autolabel(b)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig('sentiment.png', dpi=300,bbox_inches = 'tight')
    plt.show()



tokenizer = AutoTokenizer.from_pretrained('./save_models/cardiffnlp_twitter-roberta-base-sentiment', padding=True, truncation=True,max_length=512, add_special_tokens = True)
model = AutoModelForSequenceClassification.from_pretrained('./save_models/cardiffnlp_twitter-roberta-base-sentiment')
model.eval()




# data = pd.read_csv('./data/MiMic/all_data.csv')
data = pd.read_csv('./data/medical_text/all_data.csv')
 
text  = data.text.values
labels = data.labels.values
preds = []
for t in text:
    encoded_input = tokenizer(t, return_tensors='pt')
    encoded_input['input_ids'] = encoded_input['input_ids'][:,0:512]
    encoded_input['attention_mask'] = encoded_input['attention_mask'][:,0:512]
    with torch.no_grad():
        output = model(**encoded_input)
    score = output[0][0].detach().numpy()
    score = softmax(score)
    print(score)
    pred = np.argmax(score)
    preds.append(pred)
    print(pred)

preds = np.array(preds) 
human_p = preds[labels == 0]
human_d = Counter(human_p)
chatgpt_p = preds[labels == 1]
chatgpt_d = Counter(chatgpt_p)
print(human_d)
print(chatgpt_d)

human_d = [human_d[i]/len(labels[labels == 0]) for i in [0,1,2]]
chatgpt_d = [chatgpt_d[i]/len(labels[labels == 1]) for i in [0,1,2]]
human_d = [round(d *100,2) for d in human_d]
chatgpt_d = [round(d * 100,2) for d in chatgpt_d]
print(human_d)
print(chatgpt_d)

show(['negative','neutral','positive'],human_d,chatgpt_d,ylabel ='Proportion (%)', title='Sentiment comparison of medical abstract')

# 0	negative
# 1	neutral
# 2	positive
