from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import numpy as np
import random 

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

                    
stop_words = stopwords.words('english')
# print(stop_words)

def get_different(source_text,target_text):

    source_sentence_num = []
    target_sentence_num = []

    source_sentence_len = []
    target_sentenec_len = []

    source_text_len= []
    target_text_len = []

    source_words = []
    target_words = []

    source_stem_words = []
    target_stem_words = []

    for text in source_text:
        sentences = text.split('.')
        source_sentence_num.append(len(sentences))
        for sentence in sentences:
            source_sentence_len.append(len(sentence.split(' '))) 

        text_words = text.replace('.',' .').replace(',',' ,').split(' ')
        source_text_len.append(len(text_words))
        # text_words = [w for w in text_words if w not in stop_words]
        stems_words = [porter_stemmer.stem(w) for w in text_words ]
        source_words.extend(text_words)
        source_stem_words.extend(stems_words)
    
    for text in target_text:
        sentences = text.split('.')
        target_sentence_num.append(len(sentences))
        for sentence in sentences:
            target_sentenec_len.append(len(sentence.split(' '))) 

        text_words = text.replace('.',' .').replace(',',' ,').split(' ')
        target_text_len.append(len(text_words))
        # text_words = [w for w in text_words if w not in stop_words]
        stems_words = [porter_stemmer.stem(w) for w in text_words ]
        target_words.extend(text_words)
        target_stem_words.extend(stems_words)


    source_count = Counter(source_words)
    target_count = Counter(target_words)
    source_stem_count = Counter(source_stem_words)
    target_stem_count = Counter(target_stem_words)
    print('source_count:{source_count},target_count:{target_count}'.format(source_count = len(source_count),target_count =  len(target_count)))
    print('source_stem_count:{source_stem_count},target_stem_count:{target_stem_count}'.format(source_stem_count = len(source_stem_count),target_stem_count =  len(target_stem_count)))

    print('source_sentence_num:{source_sentence_num},target_sentence_num:{target_sentence_num}'.format(source_sentence_num = np.mean(source_sentence_num),target_sentence_num =  np.mean(target_sentence_num)))
    print('source_sentence_len:{source_sentence_len},target_sentenec_len:{target_sentenec_len}'.format(source_sentence_len = np.mean(source_sentence_len),target_sentenec_len =  np.mean(target_sentenec_len)))
    print('source_text_len:{source_text_len},target_text_len:{target_text_len}'.format(source_text_len = np.mean(source_text_len),target_text_len =  np.mean(target_text_len)))



    


# data = pd.read_csv('./data/MiMic/all_data.csv')
data = pd.read_csv('./data/medical_text/all_data.csv')

human_text = data[data.labels == 0].text.values
chatgpt_text = data[data.labels == 1].text.values
random.seed(2023)
chatgpt_text = random.sample(list(chatgpt_text),len(human_text))
get_different(human_text,chatgpt_text)

# source_count:22889,target_count:15782
# source_stem_count:16195,target_stem_count:11120
# source_sentence_num:8.736818181818181,target_sentence_num:10.375454545454545
# source_sentence_len:16.15196920035378,target_sentenec_len:15.670463506527645
# source_text_len:146.265,target_text_len:168.60863636363635


# source_count:11095,target_count:7733
# source_stem_count:8396,target_stem_count:5774
# source_sentence_num:12.727727272727273,target_sentence_num:12.491363636363637
# source_sentence_len:10.434448769686798,target_sentenec_len:10.19162330337324
# source_text_len:135.8840909090909,target_text_len:130.54090909090908