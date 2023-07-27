#  Differentiate ChatGPT-generated and Human-written Medical Texts


# Data Description

This paper involves two datasets: medical abstract (in data/medical_text) and radiology report (in data/MiMic) datasets.

all_ Data.csv contains all human-written data and ChatGPT-generated data

prompt*_seed*_train.csv, prompt*_seed*_val.csv, prompt*_seed*_test.csv  is the training set, validation set, and testing set for different groups.

# Software environment

```
pip install -r requirements.txt
```

# run code

1. vocabulary and sentence analysis: ```python word_count.py```
2. Part-of-speech analysis: ```python pos_analysis.py```
3. Dependency parsing: ```python dependency_analysis.py```
4. Sentiment analysis: ```python sentiment_analysis.py```
5. Text perplexity: ```python PPL_distribution.py```
6. Perplexity-CLS: ```python ppl_cls.py```
7. CART: ```python CART_cls.py```
8. XGBoost: ```python xgboost_cls.py```
9. BERT: ```python BERT_cls.py```'