#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:47:56 2022

@author: yasmin

SENTIMENT ANALYSIS WITH DEEP LEARNING USING BERT
"""
import torch
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score
import random
import optuna
import matplotlib.pyplot as plt
import pylab as pl

from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
import subprocess
#import scheduler

## Parametros##################################################################################################################################3
batch_size=4
lr=1e-5 #2e-5 > 5e-5
epochs=4
seed_val=17

## Data processing #############################################################################################################################
df=pd.read_csv("/home/yasmin/Documents/Optimization/Sentiment_Analisys/smile-annotations-final.csv",
               names=['id','text','category'])
df.set_index('id',inplace=True)
statis=df.category.value_counts()

df=df[df.category.str.contains('\|')==False]
statis=df.category.value_counts()

df=df[df.category!='nocode']
statis=df.category.value_counts()

possible_labels=df.category.unique()
label_dict={}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label]=index

df['label']=df.category.replace(label_dict)

x_train, x_val, y_train, y_val=train_test_split(df.index.values,
                                                df.label.values,
                                                test_size=0.15,
                                                random_state=17,
                                                stratify=df.label.values
                                                )
df['data_type']=['not_set']*df.shape[0]
df.loc[x_train,'data_type']='train'
df.loc[x_val,'data_type']='val'

df.groupby(['category','label','data_type']).count()

##Tokenizer and encoding data#######################################################################################################################
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',
                                        do_lower_case=True)

encoded_data_train=tokenizer.batch_encode_plus(df[df.data_type=='train'].text.values,
                                               add_special_tokens=True,
                                               return_attention_mask=True,
                                               pad_to_max_length=True,
                                               max_length=150,
                                               return_tensors='pt')

encoded_data_val=tokenizer.batch_encode_plus(df[df.data_type=='val'].text.values,
                                               add_special_tokens=True,
                                               return_attention_mask=True,
                                               pad_to_max_length=True,
                                               max_length=150,
                                               return_tensors='pt')

input_ids_train=encoded_data_train['input_ids']
attention_masks_train=encoded_data_train['attention_mask']
labels_train=torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val=encoded_data_val['input_ids']
attention_masks_val=encoded_data_val['attention_mask']
labels_val=torch.tensor(df[df.data_type=='val'].label.values)

dataset_train=TensorDataset(input_ids_train,attention_masks_train,labels_train)
dataset_val=TensorDataset(input_ids_val,attention_masks_val,labels_val)

##Modelo pre-entrenado###################################################################################################################################3
model=BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                    num_labels=len(label_dict),
                                    output_attentions=False,
                                    output_hidden_states=False)

##Métricas ##############################################################################################################################################
def f1_score_func(preds,labels):
    preds_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return f1_score(labels_flat,preds_flat,average='weighted')

def accuracy_per_class(preds,labels):
    label_dict_inverse={v: k for k, v in label_dict.items()}
    
    preds_flat=np.argmac(preds,axis=1).flatten()
    labels_flat=labels.flatten()
        
    for label in np.unique(labels_flat):
        y_preds=preds_flat[labels_flat==label]
        y_true=labels_flat[labels_flat==label]

##el loop de entrenamiento ############################################################################################################################
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device=torch.device('cpu')

def evaluate(dataloader_val):
    model.eval()
    
    loss_val_total=0
    predictions, true_vals=[],[]
    
    for batch in dataloader_val:
        batch=tuple(b.to(device) for b in batch)
        inputs={'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]}
        
        with torch.no_grad():
            outputs=model(**inputs)
        
        loss=outputs[0]
        logits=outputs[1]
        loss_val_total+=loss.item()
        
        logits=logits.detach().cpu().numpy()
        label_ids=inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        
    loss_val_avg=loss_val_total/len(dataloader_val)
    
    predictions=np.concatenate(predictions,axis=0)
    true_vals=np.concatenate(true_vals,axis=0)
    
    return loss_val_avg, predictions, true_vals

def run_train(trial,params):
    
    print("trial: ",params)
  
    ##Dataloader#################################################################################################################33333
    dataloader_train=DataLoader(dataset_train,
                                sampler=RandomSampler(dataset_train),
                                batch_size=params['batch_size'])

    dataloader_val=DataLoader(dataset_val,
                                sampler=RandomSampler(dataset_val),
                                batch_size=params['batch_size'])
    
    ##Optimizer y scheduler######################################################################################################33

    optimizer=AdamW(model.parameters(),
                    lr=params['lr'],
                    eps=1e-8)

    scheduler=get_linear_schedule_with_warmup(optimizer, 
                                             num_warmup_steps=0, 
                                             num_training_steps=len(dataloader_train)*params['epochs'])
    ##Train######################################################################################################################
    for epoch in tqdm(range(1,params['epochs']+1)):
        model.train()
        loss_train_total=0
        progress_bar=tqdm(dataloader_train,
                          desc='Epoch {:1d}'.format(epoch),
                          leave=False,
                          disable=False)
        
        for batch in progress_bar:
            model.zero_grad()
            
            batch=tuple(b.to(device) for b in batch)
            inputs={'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]}
        
            outputs=model(**inputs)
            loss=outputs[0]
            loss_train_total+=loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
        torch.save(model.state_dict(), '/home/yasmin/Documents/Optimization/Sentiment_Analisys/Models/BERT_ft_epoch{}.model'.format(params['epochs']))
        tqdm.write('\nEpoch {}'.format(epoch))
        loss_train_avg=loss_train_total/len(dataloader_train)
        
        tqdm.write('Training loss: {}'.format(loss_train_avg))
        
        val_loss, predictions, true_vals=evaluate(dataloader_val)
        val_f1=f1_score_func(predictions,true_vals)
        tqdm.write('Validation loss: {}'.format(val_loss))
        tqdm.write('F1 Score (weighted: {}'.format(val_f1))
        loss_val_avg=val_loss/len(dataloader_val)
        
        val_loss, predictions, true_vals=evaluate(dataloader_val)
        val_f1=f1_score_func(predictions,true_vals)
        
        #Pruning optuna                                                          
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return loss_val_avg #val para validación, no usar el set de train

## Optuna ##########################################################################################################################
def objective(trial):
    params={"batch_size": trial.suggest_int("batch_size",1,5),
            "lr": trial.suggest_loguniform("lr",2e-5,5e-5),
            "epochs": trial.suggest_int("epochs",2,4)}
    
    all_losses=[]
    temp_loss=run_train(trial,params)
    all_losses.append(temp_loss)
    

    return np.mean(all_losses)
        
n_trials=5
study=optuna.create_study(direction="minimize")
study.optimize(objective,n_trials=n_trials)

##Prunning
pruned_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.PRUNED]
complete_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]

##Save log
file_ = open('/home/yasmin/Documents/Optimization/Sentiment_Analisys/log.txt', 'w+') 
subprocess.run('echo Hello from shell', shell=True, stdout=file_) 
file_.close() 

##Print
print("Study statistics: ")
print("Number of finished trials: ",len(study.trials))
print("number of pruned trials: ",len(pruned_trials))
print("Number of complete trials: ",len(complete_trials))

trial_=study.best_trial
print("best trial: ",trial_.values)
print("best paraneters: ",trial_.params)

## Graficando resultados ###############################################################################################################
trials=study.trials
print(trials[0].values[0])

"""
y=[]
t=[]
for i in range(0,n_trials):
    #print(i)
    y=np.append(y,trials[i].values[0])
    t=np.append(t,i)
    
pl.figure(figsize=(8, 6), dpi=80)
pl.subplot(1, 1, 1)
pl.plot(t, y, color="blue", linewidth=1.0, linestyle="o")

for i in range(0,n_trials):
    pl.annotate(trials[i].params,
            xy=(i, y[i]), xycoords='data',
            xytext=(+10, +30), textcoords='offset points', fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
pl.show()
"""

##Visualize the optimization history
plot_optimization_history(study)

##Visualize the learning curves of the trials
#plot_intermediate_values(study)

##Visualize high-dimensional parameter relationships
plot_parallel_coordinate(study)

##Visualize hyperparameter relationships
plot_contour(study)

##Visualize individual hyperparameters as slice plot.
plot_slice(study)

##Visualize parameter importances
plot_param_importances(study)

##Learn which hyperparameters are affecting the trial duration with hyperparameter importance.
optuna.visualization.plot_param_importances(
    study, target=lambda t: t.duration.total_seconds(), target_name="duration"
)


scores=0

    
"""
##Cargando y evaluando el modelo##############################################
model=BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                num_labels=len(label_dict),
                                output_attentions=False,
                                output_hidden_states=False)  
model.load_state_dict(torch.load('Models/finetuned_bert_epoch_1_gpu_trained.model',map_location=torch.device('cpu')))

_, predictions, true_vals=evaluate(dataloader_val)
accuracy_per_class(predictions,true_vals)
"""