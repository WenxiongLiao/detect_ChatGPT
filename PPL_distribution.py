import seaborn
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import seaborn as sns
import random

from GPTZero_model import GPT2PPL

plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 12})



model = GPT2PPL(model_id="./save_models/microsoft_biogpt")
human_dist = []
chatgpt_dist = []
# data = pd.read_csv('./data/MiMic/all_data.csv')
data = pd.read_csv('./data/medical_text/all_data.csv')
samples = data.text.values
labels = data.labels.values
preds = []
for (sample,label) in zip(samples,labels):
    ppl = model.getPPL(sample)
    if label == 0:
        human_dist.append(ppl)
    elif label == 1:
        chatgpt_dist.append(ppl)

random.seed(2013)
chatgpt_dist = random.sample(chatgpt_dist,len(human_dist))

labels = np.concatenate([['human'] * len(human_dist),['ChatGPT'] * len(chatgpt_dist)])
dist = human_dist
dist.extend(chatgpt_dist)
data = pd.DataFrame({'Perplexity':dist,'labels':labels})
print(data.shape)
sns.displot(data=data, x="Perplexity", kind="kde", hue="labels",palette = {'human':'blue','ChatGPT':'#ed7857'},height=3)

plt.savefig("PPL.jpg",dpi=300)