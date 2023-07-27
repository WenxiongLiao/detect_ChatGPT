from datasets import load_dataset,Dataset,DatasetDict
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel,AutoTokenizer,AutoModel,AutoModelForSequenceClassification,AutoConfig,DataCollatorWithPadding,BertForSequenceClassification
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score,classification_report
import itertools
import random
import os



def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True,max_length = max_length)

def evalate(dataloader,model,device):
    ground_truth = []
    preds = []
    
    model.eval()
    for i,batch in enumerate(dataloader):
        
        
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        labels = labels.detach().cpu().numpy()
        
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits,dim = -1).detach().cpu().numpy()
        
        if len(ground_truth) == 0:
            ground_truth = labels
            preds = pred
        else:
            ground_truth = np.concatenate([ground_truth,labels])
            preds = np.concatenate([preds,pred])
#     print(ground_truth)
#     print(preds)
    acc = accuracy_score(y_true = ground_truth, y_pred = preds)
    print(classification_report(y_true = ground_truth, y_pred = preds,digits=3))
    
    return acc

for prompt in [1,2]:
    for seed in [1,2]:
        # train_path = f'./data/MiMic/prompt{prompt}_seed{seed}_train.csv'
        # val_path = f'./data/MiMic/prompt{prompt}_seed{seed}_val.csv'
        # test_path = f'./data/MiMic/prompt{prompt}_seed{seed}_test.csv'
        # save_dir = f"./save_models/MiMic_bert-base-cased_prompt{prompt}_seed{seed}"

        train_path = f'./data/medical_text/prompt{prompt}_seed{seed}_train.csv'
        test_path = f'./data/medical_text/prompt{prompt}_seed{seed}_test.csv'
        val_path = f'./data/medical_text/prompt{prompt}_seed{seed}_val.csv'
        save_dir = f"./save_models/medical_text_bert-base-cased_prompt{prompt}_seed{seed}"

        seed_everything(1234)
        device = torch.device("cuda:6")
        model_checkpoint = "./save_models/bert-base-cased"
        max_length = 512
        epochs = 5
        batch_size = 8
        lr = 5e-5

        

        # 0为hunman
        config = AutoConfig.from_pretrained(model_checkpoint, label2id={'human':0,'chatgpt':1}, id2label={0:'human',1:'chatgpt'})
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding=True, truncation=True,model_max_length = max_length)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


        raw_datasets = load_dataset("csv", data_files={"train":train_path,"val":val_path,"test":test_path})

        tokenized_datasets = raw_datasets.map(
            tokenize_function, batched=True, remove_columns= ['text']
        )

        # tokenized_datasets = tokenized_datasets["train"].train_test_split(seed = 2023, test_size=0.3)


        model = BertForSequenceClassification.from_pretrained(
                        model_checkpoint, config=config)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5)



        train_dataloader = DataLoader(
                    tokenized_datasets["train"], batch_size=batch_size, collate_fn = data_collator, shuffle=True)
        val_dataloader = DataLoader(
                    tokenized_datasets["val"], batch_size=batch_size, collate_fn = data_collator)
        test_dataloader = DataLoader(
            tokenized_datasets["test"], batch_size=batch_size, collate_fn = data_collator)

        max_acc = 0
        for epoch in range(epochs):
            model.train()
            for i, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
        #             labels = batch.pop('labels')
                outputs = model(**batch, output_hidden_states  = True)
                loss = outputs.loss
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            acc = evalate(val_dataloader,model,device)
            if acc > max_acc:
                print('{max_acc}===>>{acc}'.format(max_acc = max_acc, acc = acc))
                max_acc = acc
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

        model = BertForSequenceClassification.from_pretrained(
                        save_dir, config=config)
        model.to(device)
        acc = evalate(test_dataloader,model,device)
        print(f'acc:{acc}')




# medical_text
#               precision    recall  f1-score   support

#            0      0.993     0.970     0.982       440
#            1      0.971     0.993     0.982       440

#     accuracy                          0.982       880
#    macro avg      0.982     0.982     0.982       880
# weighted avg      0.982     0.982     0.982       880

#               precision    recall  f1-score   support

#            0      0.988     0.975     0.982       440
#            1      0.975     0.989     0.982       440

#     accuracy                          0.982       880
#    macro avg      0.982     0.982     0.982       880
# weighted avg      0.982     0.982     0.982       880


#               precision    recall  f1-score   support

#            0      0.988     0.966     0.977       440
#            1      0.967     0.989     0.978       440

#     accuracy                          0.977       880
#    macro avg      0.978     0.977     0.977       880
# weighted avg      0.978     0.977     0.977       880


#               precision    recall  f1-score   support

#            0      0.993     0.980     0.986       440
#            1      0.980     0.993     0.986       440

#     accuracy                          0.986       880
#    macro avg      0.986     0.986     0.986       880
# weighted avg      0.986     0.986     0.986       880


# acc: [0.982,0.982,0.977,0.986] 0.982
# precision: [0.982,0.982,0.978,0.986]  0.982
# recall: [0.982,0.982,0.977,0.986]  0.982
# f1: [0.982,0.982,0.977,0.986] 0.982



# MiMic
#               precision    recall  f1-score   support

#            0      0.981     0.961     0.971       440
#            1      0.962     0.982     0.972       440

#     accuracy                          0.972       880
#    macro avg      0.972     0.972     0.972       880
# weighted avg      0.972     0.972     0.972       880


#               precision    recall  f1-score   support

#            0      0.979     0.977     0.978       440
#            1      0.977     0.980     0.978       440

#     accuracy                          0.978       880
#    macro avg      0.978     0.978     0.978       880
# weighted avg      0.978     0.978     0.978       880


#               precision    recall  f1-score   support

#            0      0.965     0.989     0.976       440
#            1      0.988     0.964     0.976       440

#     accuracy                          0.976       880
#    macro avg      0.976     0.976     0.976       880
# weighted avg      0.976     0.976     0.976       880


#               precision    recall  f1-score   support

#            0      0.873     0.934     0.902       440
#            1      0.929     0.864     0.895       440

#     accuracy                          0.899       880
#    macro avg      0.901     0.899     0.899       880
# weighted avg      0.901     0.899     0.899       880

# acc: [0.972,0.978,0.976,0.899] 0.95625
# precision: [0.972,0.978,0.976,0.901] 0.95675
# recall:[0.972,0.978,0.976 ,0.899] 0.95625
# f1: [0.972,0.978,0.976,0.899 ] 0.95625


