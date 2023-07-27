


import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import xgboost as xgb
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn import tree
import graphviz
import numpy as np

plt.rc('font',family='Times New Roman')

def processing_sentence(x, stop_words):
    cut_word = x.split()
    words = [word for word in cut_word if word not in stop_words and word != ' ']
    return ' '.join(words)


def data_processing(train_path,test_path):


    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.text
    y_train = train_data.labels
    x_test = test_data.text
    y_test = test_data.labels
    

    stop_words = stopwords.words('english')
    x_train = x_train.apply(lambda x: processing_sentence(x, stop_words))
    x_test = x_test.apply(lambda x: processing_sentence(x, stop_words))

    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    x_train_weight = x_train.toarray()
    x_test_weight = x_test.toarray()

    wor2idx = tf.vocabulary_
    idx2word = {idx:word for word,idx in wor2idx.items()}
    feature_names = []
    for i in range(len(idx2word)):
        feature_names.append(idx2word[i])


    return x_train_weight, x_test_weight, y_train, y_test,feature_names

for prompt in [1,2]:
    for seed in [1,2]:
        # train_path = f'./data/MiMic/prompt{prompt}_seed{seed}_train.csv'
        # test_path = f'./data/MiMic/prompt{prompt}_seed{seed}_test.csv'

        train_path = f'./data/medical_text/prompt{prompt}_seed{seed}_train.csv'
        test_path = f'./data/medical_text/prompt{prompt}_seed{seed}_test.csv'



        x_train_weight, x_test_weight, y_train, y_test,feature_names = data_processing(train_path,test_path)
        clf = tree.DecisionTreeClassifier(random_state=0,max_depth=4)


        clf = clf.fit(x_train_weight, y_train)

        # plt.figure(figsize=(14,14))
        # tree.plot_tree(clf,feature_names=feature_names,max_depth=4, fontsize=7,class_names=['human','chatgpt'],filled= True)
        # plt.show()
        # plt.savefig('tree_high_dpi', dpi=300)

        dot_data = tree.export_graphviz(clf, out_file=None, 
                                        feature_names=feature_names,  
                                        class_names=['human','chatgpt'],
                                        filled=True,
                                        rounded = True)

        # Draw graph
        graph = graphviz.Source(dot_data, format="png") 
        graph.render(f"decision_tree_graphivz_{prompt}_{seed}")
        y_predict = clf.predict(x_test_weight)

        confusion_mat = metrics.confusion_matrix(y_test, y_predict)
        print('准确率：', metrics.accuracy_score(y_test, y_predict))
        print("confusion_matrix is: ", confusion_mat)
        print('分类报告:', metrics.classification_report(y_test, y_predict,digits=3))

        # dot -Tpng decision_tree_graphivz -o decision_tree_graphivz.png


# medical_text
# 准确率： 0.8829545454545454
# confusion_matrix is:  [[428  12]
#  [ 91 349]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.825     0.973     0.893       440
#            1      0.967     0.793     0.871       440

#     accuracy                          0.883       880
#    macro avg      0.896     0.883     0.882       880
# weighted avg      0.896     0.883     0.882       880

# 准确率： 0.8875
# confusion_matrix is:  [[429  11]
#  [ 88 352]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.830     0.975     0.897       440
#            1      0.970     0.800     0.877       440

#     accuracy                          0.887       880
#    macro avg      0.900     0.887     0.887       880
# weighted avg      0.900     0.887     0.887       880

# 准确率： 0.8647727272727272
# confusion_matrix is:  [[431   9]
#  [110 330]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.797     0.980     0.879       440
#            1      0.973     0.750     0.847       440

#     accuracy                          0.865       880
#    macro avg      0.885     0.865     0.863       880
# weighted avg      0.885     0.865     0.863       880

# 准确率： 0.8397727272727272
# confusion_matrix is:  [[431   9]
#  [132 308]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.766     0.980     0.859       440
#            1      0.972     0.700     0.814       440

#     accuracy                          0.840       880
#    macro avg      0.869     0.840     0.837       880
# weighted avg      0.869     0.840     0.837       880

# acc: [0.883, 0.887,0.865,0.840 ] 0.8687499999999999
# precision: [0.896,0.900,0.885,0.869] 0.8875
# recall:[0.883,0.887,0.86,0.840] 0.8674999999999999
# f1: [0.882,0.887,0.863,0.837 ]0.867250000



# MiMic
# 准确率： 0.8363636363636363
# confusion_matrix is:  [[357  83]
#  [ 61 379]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.854     0.811     0.832       440
#            1      0.820     0.861     0.840       440

#     accuracy                          0.836       880
#    macro avg      0.837     0.836     0.836       880
# weighted avg      0.837     0.836     0.836       880

# 准确率： 0.8306818181818182
# confusion_matrix is:  [[346  94]
#  [ 55 385]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.863     0.786     0.823       440
#            1      0.804     0.875     0.838       440

#     accuracy                          0.831       880
#    macro avg      0.833     0.831     0.830       880
# weighted avg      0.833     0.831     0.830       880

# 准确率： 0.8238636363636364
# confusion_matrix is:  [[395  45]
#  [110 330]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.782     0.898     0.836       440
#            1      0.880     0.750     0.810       440

#     accuracy                          0.824       880
#    macro avg      0.831     0.824     0.823       880
# weighted avg      0.831     0.824     0.823       880

# 准确率： 0.8318181818181818
# confusion_matrix is:  [[413  27]
#  [121 319]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.773     0.939     0.848       440
#            1      0.922     0.725     0.812       440

#     accuracy                          0.832       880
#    macro avg      0.848     0.832     0.830       880
# weighted avg      0.848     0.832     0.830       880

# acc: [0.836,0.831,0.824 ,0.832] 0.8307499999999999
# precision: [0.837,0.833,0.831,0.848] 0.8372499999999999
# recall: [0.836,0.831,0.824,0.832] 0.8307499999999999
# f1: [0.836,0.830,0.823,0.830] 0.82975
