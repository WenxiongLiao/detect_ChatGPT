
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# stop_words = stopwords.words('english')
plt.rc('font',family='Times New Roman')

def get_pos_proportion(data_path, human_label = 0, chatgpt_label = 1, top_N = 20):

    data = pd.read_csv(data_path)
    human_text = data[data.labels == human_label].text.values
    chatgpt_text = data[data.labels == chatgpt_label].text.values

    pos_list = []

    for t in human_text:
        t_pos = pos_tag(word_tokenize(t))
        for word,pos in t_pos:
            if word not in [',','.','(',')',':']:
                pos_list.append(pos)
    
    pos_counts = Counter(pos_list)
    top_pos = pos_counts.most_common(top_N)
    top_pos_pro = []
    for pos,fre in top_pos:
        top_pos_pro.append([pos,fre/len(pos_list) * 100])


    cahtgpt_pos_list = []
    for t in chatgpt_text:
        t_pos = pos_tag(word_tokenize(t))
        for word,pos in t_pos:
            if word not in [',','.','(',')',':']:
                cahtgpt_pos_list.append(pos)
    chatgpt_pos_counts = Counter(cahtgpt_pos_list)
 

    chatgpt_pos_counts = {pos:fre/len(cahtgpt_pos_list) * 100   for pos, fre in chatgpt_pos_counts.items()}

    pos_list = []
    human_distribution = []
    chatgpt_distribution = []
    for pos,pro in top_pos_pro:
        pos_list.append(pos)
        human_distribution.append(pro)
        if pos in chatgpt_pos_counts.keys():
            chatgpt_distribution.append(chatgpt_pos_counts[pos])
        else:
            chatgpt_distribution.append(0)


    human_distribution = [round(d,1) for d in human_distribution]
    chatgpt_distribution = [round(d,1) for d in chatgpt_distribution]

    # print(np.sum(human_distribution))
    # print(np.sum(chatgpt_distribution))
    return pos_list,human_distribution,chatgpt_distribution

#定义函数来显示柱状上的数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

def show(sub_label,x_bar,y_bar,ylabel,title):
    total_width, n = 1, 3
    width = total_width / n
    x= list(range(top_N))
    plt.rc('font', size=12)
    plt.figure(figsize=(10, 2))
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
    plt.savefig('pos_proportion.png', dpi=300,bbox_inches = 'tight')
    plt.show()

#     total_width, n = 1, 3
#     width = total_width / n
#     x= list(range(top_N))
#     plt.rc('font', size=7)
#     plt.figure(figsize=(13, 4))

#     # fig, ax=plt.subplots()
# #     ax.spines['top'].set_visible(False)
# #     ax.spines['right'].set_visible(False)
#     # ax.spines['bottom'].set_visible(False)
#     # ax.spines['left'].set_visible(False)

  
#     a = plt.bar(x , x_bar, color = '#aadfff', width = width, label='human',tick_label = sub_label)
#     for i in range(len(x)):
#         x[i] = x[i] + width
#     b = plt.bar(np.array(x) + width , y_bar, color = '#8fd49e', width = width, label='chatgpt')
# #     autolabel(a)
# #     autolabel(b)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.savefig('pos_proportion.png', dpi=300,bbox_inches = 'tight')
#     plt.show()

# data = './data/MiMic/all_data.csv'
data = './data/medical_text/all_data.csv'

top_N = 20
pos_list,human_distribution,chatgpt_distribution = get_pos_proportion(data,top_N = top_N)
show(pos_list,human_distribution,chatgpt_distribution,ylabel ='Proportion (%)', title='Part-of-Speech comparison of radiology report')


# https://blog.csdn.net/weixin_43720526/article/details/120548774




