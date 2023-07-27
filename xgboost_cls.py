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
import numpy  as np

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
        start = time.time()
        print("start time is: ", start)
        model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=50, n_jobs=2,
                                    silent=False, objective='binary:logistic')
        model.fit(x_train_weight, y_train)
        model.get_booster().feature_names = feature_names
        end = time.time()
        print("end time is: ", end)
        print("cost time is: ", (end - start))
        y_predict = model.predict(x_test_weight)

        confusion_mat = metrics.confusion_matrix(y_test, y_predict)
        print('准确率：', metrics.accuracy_score(y_test, y_predict))
        print("confusion_matrix is: ", confusion_mat)
        print('分类报告:', metrics.classification_report(y_test, y_predict,digits=3))

        # fig, ax = plt.subplots(figsize=(15, 15))
        plt.rcParams["figure.figsize"] = (5, 5)
        plot_importance(model,
                        max_num_features=15,
                        height = 0.3,
                        grid=False,
                        show_values= False
                        )

        plt.show()



# medical_text
# 准确率： 0.9625
# confusion_matrix is:  [[431   9]
#  [ 24 416]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.947     0.980     0.963       440
#            1      0.979     0.945     0.962       440

#     accuracy                          0.963       880
#    macro avg      0.963     0.962     0.962       880
# weighted avg      0.963     0.963     0.962       880

# 准确率： 0.9613636363636363
# confusion_matrix is:  [[433   7]
#  [ 27 413]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.941     0.984     0.962       440
#            1      0.983     0.939     0.960       440

#     accuracy                          0.961       880
#    macro avg      0.962     0.961     0.961       880
# weighted avg      0.962     0.961     0.961       880

# 准确率： 0.9590909090909091
# confusion_matrix is:  [[431   9]
#  [ 27 413]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.941     0.980     0.960       440
#            1      0.979     0.939     0.958       440

#     accuracy                          0.959       880
#    macro avg      0.960     0.959     0.959       880
# weighted avg      0.960     0.959     0.959       880

# 准确率： 0.9454545454545454
# confusion_matrix is:  [[431   9]
#  [ 39 401]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.917     0.980     0.947       440
#            1      0.978     0.911     0.944       440

#     accuracy                          0.945       880
#    macro avg      0.948     0.945     0.945       880
# weighted avg      0.948     0.945     0.945       880

# acc: [0.963,0.961,0.959,0.945] 0.957
# precision: [0.963,0.962,0.960,0.948] 0.9582499999999999
# recall: [0.962,0.961,0.959,0.945] 0.95675
# f1: [0.962,0.961,0.959, 0.945] 0.95675





# MiMic
# 准确率： 0.928409090909091
# confusion_matrix is:  [[417  23]
#  [ 40 400]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.912     0.948     0.930       440
#            1      0.946     0.909     0.927       440

#     accuracy                          0.928       880
#    macro avg      0.929     0.928     0.928       880
# weighted avg      0.929     0.928     0.928       880


# 准确率： 0.9261363636363636
# confusion_matrix is:  [[416  24]
#  [ 41 399]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.910     0.945     0.928       440
#            1      0.943     0.907     0.925       440

#     accuracy                          0.926       880
#    macro avg      0.927     0.926     0.926       880
# weighted avg      0.927     0.926     0.926       880


# 准确率： 0.928409090909091
# confusion_matrix is:  [[419  21]
#  [ 42 398]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.909     0.952     0.930       440
#            1      0.950     0.905     0.927       440

#     accuracy                          0.928       880
#    macro avg      0.929     0.928     0.928       880
# weighted avg      0.929     0.928     0.928       880


# 准确率： 0.9125
# confusion_matrix is:  [[418  22]
#  [ 55 385]]
# 分类报告:               precision    recall  f1-score   support

#            0      0.884     0.950     0.916       440
#            1      0.946     0.875     0.909       440

#     accuracy                          0.912       880
#    macro avg      0.915     0.912     0.912       880
# weighted avg      0.915     0.912     0.912       880

# acc: [0.928,0.926,0.928,0.912]  0.9235
# precision: [0.929,0.927,0.929,0.915] 0.925
# recall: [0.928,0.926,0.928,0.912] 0.9235
# f1: [0.928,0.926,0.928,0.912] 0.9235
