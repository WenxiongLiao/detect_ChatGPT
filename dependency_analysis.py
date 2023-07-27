from stanfordcorenlp import StanfordCoreNLP 
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np




plt.rc('font',family='Times New Roman')

def get_dependency_proportion(data_path, human_label = 0, chatgpt_label = 1,top_N = 20):

    data = pd.read_csv(data_path)

    human_text = data[data.labels == human_label].text.values
    chatgpt_text = data[data.labels == chatgpt_label].text.values

    dependency_list = []
    for t in human_text:
        ret = nlp.dependency_parse(t) 
        for denpend,head,tail in ret:
            dependency_list.append([denpend,tail - head])
    
    # print(dependency_list)

    dependency_counts = Counter(np.array(dependency_list)[:,0])
    top_dependency = dependency_counts.most_common(top_N)
    # print(top_dependency)
    top_dependency_pro = []
    top_dependency_dis = []
    for dependency,fre in top_dependency:
        top_dependency_pro.append([dependency,fre/len(dependency_list) * 100])

        dists = []
        for depend,dist in dependency_list:
            if depend == dependency:
                dists.append(abs(dist))
        top_dependency_dis.append([dependency,0] if len(dists)==0 else [dependency,np.mean(dists)])



    chatgpt_dependency_list = []
    for t in chatgpt_text:
        ret = nlp.dependency_parse(t) 
        for denpend,head,tail in ret:
            chatgpt_dependency_list.append([denpend,tail - head])
    chatgpt_dependency_counts = Counter(np.array(chatgpt_dependency_list)[:,0])
    chatgpt_dependency_counts = {pos:fre/len(chatgpt_dependency_list) * 100   for pos, fre in chatgpt_dependency_counts.items()}

    dependency_list = []
    human_distribution = []
    chatgpt_distribution = []
    for dependency,pro in top_dependency_pro:
       dependency_list.append(dependency)
       human_distribution.append(pro)
       if dependency in chatgpt_dependency_counts.keys():
          chatgpt_distribution.append(chatgpt_dependency_counts[dependency])
       else:
          chatgpt_distribution.append(0)

    human_distance = []
    chatgpt_distance = []
    for dependency,dis in top_dependency_dis:
        human_distance.append(dis)
        dists = []
        for depend,dist in chatgpt_dependency_list:
            if depend == dependency:
                dists.append(abs(dist))
        chatgpt_distance.append(0 if len(dists)==0 else np.mean(dists))


    human_distribution = [round(d,1) for d in human_distribution]
    chatgpt_distribution = [round(d,1) for d in chatgpt_distribution]
    print(np.sum(human_distribution))
    print(np.sum(chatgpt_distribution))

    human_distance = [round(d,1) for d in human_distance]
    chatgpt_distance = [round(d,1) for d in chatgpt_distance]



    

    return dependency_list,human_distribution,chatgpt_distribution,human_distance,chatgpt_distance


#定义函数来显示柱状上的数值
def autolabel(rects):
 for rect in rects:
  height = rect.get_height()
  plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))


def show(sub_label,x_bar,y_bar,ylabel,title,save_path):
    total_width, n = 1, 3
    width = total_width / n
    x= list(range(len(sub_label)))
    plt.rc('font', size=11)
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
    plt.savefig(save_path, dpi=300,bbox_inches = 'tight')
    plt.show()


top_N = 20
nlp = StanfordCoreNLP('./stanford-corenlp-full-2018-10-05') 
data_path = './data/medical_text/all_data.csv'
# data_path = './data/MiMic/all_data.csv'

dependency_list,human_distribution,chatgpt_distribution,human_distance,chatgpt_distance = get_dependency_proportion(data_path, human_label = 0, chatgpt_label = 1,top_N = top_N)

show(dependency_list,human_distribution,chatgpt_distribution,ylabel ='Proportion (%)', title='Dependency comparison of radiology report',save_path = 'dependency_proportion.png')
show(dependency_list,human_distance,chatgpt_distance,ylabel ='Distance', title='Dependency distance comparison of radiology report',save_path = 'dependency_distance.png')

# https://blog.csdn.net/weixin_44826203/article/details/121253732