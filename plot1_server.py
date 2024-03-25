import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

color1 = (66/255 ,122/255 ,178/255 )
color2 = (240/255 ,145/255 ,72/255 )
cmap_color1 = [(1, 1, 1), (66/255 ,122/255 ,178/255 )] 
cmap_color2 = [(1, 1, 1), (240/255 ,145/255 ,72/255 )] 

res=np.load("All_r5.npy",allow_pickle=True) #Name of result from Optimization.py
Sn0=res[0]
S=res[1]
Test=res[3]
Importance=res[4]
dis_sub_mean=res[5]
for i in range(len(Sn0)):
    Sn0[i]=Sn0[i].item()

def approximate():
    a=np.array(Sn0)
    b=np.array(S)
    print(np.max(np.abs(a-b)))

def fig1(): 
    n=len(S)
    x=list(range(1,n+1))
    y1=list(np.array(S)*100)
    y2=list(np.array(Test)*100)
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x=x, y=y1, label='Training accuracy', color=color1)
    sns.lineplot(x=x, y=y2, label='Testing accuracy', color=color2)
    #ax.set_title('The trend of the training and testing accuracy', fontsize=18)
    ax.set_xlabel('Iterations', fontsize=15)
    ax.set_ylabel('Accuracy (%)', fontsize=15)
    plt.xticks([0,50,100,150,200])  
    plt.yticks([60,70,80,90]) 
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    plt.show()

def fig_weight(): #weight
    L=len(Importance)
    Imp=Importance[L-50]
    W=np.zeros((3,9))
    for i in range(0,9):
        for j in range(0,3):
            W[j][i]=Imp[3*i+j]/dis_sub_mean[3*i+j]
    tem=np.max(W)
    for i in range(0,9):
        for j in range(0,3):
            W[j][i]/=tem
    df = pd.DataFrame(W)
    df=df.rename(columns={i:i+1 for i in range(9)})
    sns.set(style="whitegrid", font_scale=0.5)
    sns.heatmap(data=df, annot=True, cmap="Reds",linewidths=0.5, square=True,xticklabels=df.columns, yticklabels=df.index, cbar_kws={'shrink': 0.6}, cbar=False)
    plt.savefig('C:\\Users\\11279\\Desktop\\bio\\my_paper\\weight_training\\new\\MDPI\\fig_W.eps', dpi=300, bbox_inches='tight')
    plt.show()

def fig_importance(): #importance 
    L=len(Importance)
    Imp=Importance[L-50]
    Imp=Imp/max(Imp)
    I=np.zeros((3,9))
    for i in range(0,9):
        for j in range(0,3):
            I[j][i]=Imp[3*i+j]  
    df = pd.DataFrame(I)
    df=df.rename(columns={i:i+1 for i in range(9)})
    sns.set(style="whitegrid", font_scale=0.5)
    sns.heatmap(data=df, annot=True, cmap="Blues",linewidths=0.5, square=True,xticklabels=df.columns, yticklabels=df.index, cbar_kws={'shrink': 0.6}, cbar=False)
    plt.savefig('C:\\Users\\11279\\Desktop\\bio\\my_paper\\weight_training\\new\\MDPI\\fig_I.eps', dpi=300, bbox_inches='tight')
    plt.show()

