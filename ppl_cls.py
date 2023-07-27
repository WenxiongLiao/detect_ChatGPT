import seaborn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report

from GPTZero_model import GPT2PPL




def get_best_split(human_dist,chatgpt_dist):

    human_dist = np.array(human_dist)
    chatgpt_dist = np.array(chatgpt_dist)

    max_acc = 0
    best_th = 5

    total_num = len(human_dist) + len(chatgpt_dist)

    for th in range(5,100,5):
        num = len(human_dist[human_dist>th])
        num = num + len(chatgpt_dist[chatgpt_dist<=th])
        acc = num / total_num

        if acc > max_acc:
            max_acc = acc
            best_th = th

    print(best_th,max_acc)

    return best_th





model = GPT2PPL(model_id="./save_models/microsoft_biogpt")



for prompt in [1,2]:
    for seed in [1,2]:

        human_dist = []
        chatgpt_dist = []
        # data = pd.read_csv(f'./data/medical_text/prompt{prompt}_seed{seed}_val.csv')
        data = pd.read_csv(f'./data/MiMic/prompt{prompt}_seed{seed}_test.csv')
        samples = data.text.values
        labels = data.labels.values
        preds = []
        for (sample,label) in zip(samples,labels):
            ppl = model.getPPL(sample)
            if label == 0:
                human_dist.append(ppl)
            elif label == 1:
                chatgpt_dist.append(ppl)
        th = get_best_split(human_dist,chatgpt_dist)


        human_dist = []
        chatgpt_dist = []
        data = pd.read_csv(f'./data/medical_text/prompt{prompt}_seed{seed}_test.csv')
        # data = pd.read_csv(f'./data/MiMic/prompt{prompt}_seed{seed}_test.csv')

        samples = data.text.values
        labels = data.labels.values
        preds = []
        for (sample,label) in zip(samples,labels):
            ppl = model.getPPL(sample)
            if label == 0:
                human_dist.append(ppl)
            elif label == 1:
                chatgpt_dist.append(ppl)

        human_dist = np.array(human_dist)
        chatgpt_dist = np.array(chatgpt_dist)
        labels = np.concatenate([[0] * len(human_dist), [1] * len(chatgpt_dist)] )

        human_pred = np.array([0] * len(human_dist))
        human_pred[human_dist<=th] = 1
        chatgpt_pred = np.array([1] * len(chatgpt_dist))
        chatgpt_pred[chatgpt_dist>th] = 0
        preds = np.concatenate([human_pred,chatgpt_pred])

        print(classification_report(y_true= labels,y_pred=preds,digits=3))


# medical_text
# 10 0.8636363636363636
#               precision    recall  f1-score   support

#            0      0.886     0.814     0.848       440
#            1      0.828     0.895     0.860       440

#     accuracy                          0.855       880
#    macro avg      0.857     0.855     0.854       880
# weighted avg      0.857     0.855     0.854       880

# 10 0.85
#               precision    recall  f1-score   support

#            0      0.909     0.814     0.859       440
#            1      0.831     0.918     0.873       440

#     accuracy                          0.866       880
#    macro avg      0.870     0.866     0.866       880
# weighted avg      0.870     0.866     0.866       880

# 10 0.8272727272727273
#               precision    recall  f1-score   support

#            0      0.850     0.814     0.832       440
#            1      0.821     0.857     0.839       440

#     accuracy                          0.835       880
#    macro avg      0.836     0.835     0.835       880
# weighted avg      0.836     0.835     0.835       880

# 10 0.8386363636363636
#               precision    recall  f1-score   support

#            0      0.846     0.814     0.830       440
#            1      0.821     0.852     0.836       440

#     accuracy                          0.833       880
#    macro avg      0.833     0.833     0.833       880
# weighted avg      0.833     0.833     0.833       880

# acc: [0.855,0.866,0.835,0.833] 0.8472500000000001
# precision: [0.857,0.870,0.836,0.833] 0.849
# recall: [0.855,0.866,0.835,0.833] 0.8472500000000001
# f1: [0.854,0.866,0.835,0.833] 0.847



# MiMic
# 55 0.7522727272727273
#               precision    recall  f1-score   support

#            0      0.830     0.634     0.719       440
#            1      0.704     0.870     0.778       440

#     accuracy                          0.752       880
#    macro avg      0.767     0.752     0.749       880
# weighted avg      0.767     0.752     0.749       880

# 55 0.7568181818181818
#               precision    recall  f1-score   support

#            0      0.840     0.634     0.723       440
#            1      0.706     0.880     0.783       440

#     accuracy                          0.757       880
#    macro avg      0.773     0.757     0.753       880
# weighted avg      0.773     0.757     0.753       880

# 55 0.7306818181818182
#               precision    recall  f1-score   support

#            0      0.786     0.634     0.702       440
#            1      0.693     0.827     0.754       440

#     accuracy                          0.731       880
#    macro avg      0.740     0.731     0.728       880
# weighted avg      0.740     0.731     0.728       880

# 55 0.7329545454545454
#               precision    recall  f1-score   support

#            0      0.790     0.634     0.704       440
#            1      0.694     0.832     0.757       440

#     accuracy                          0.733       880
#    macro avg      0.742     0.733     0.730       880
# weighted avg      0.742     0.733     0.730       880

# acc: [0.752,0.757,0.731,0.733] 0.74325
# precision: [0.767,0.773,0.740,0.742] 0.7555000000000001
# recall: [0.752,0.757,0.731,0.733] 0.74325
# f1: [0.749,0.753,0.728,0.730] 0.74

