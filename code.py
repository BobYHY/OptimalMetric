import torch
import numpy as np
import random
import torch.nn.functional as F
import sys

K=9
O=2
size=11559
training_percentage=0.8
P=50
lr=0.1
torch.set_default_dtype(torch.float64)

family=np.load("data\\family_clean.npy")
dis=np.zeros(((O+1)*K,size,size))
for i in range(0,K):
    for j in range(0,(O+1)):
        tem=np.load('data\\Order_dis'+str(i+1)+'_'+str(j)+'.npy')
        dis[(O+1)*i+j,:,:]=tem
print("read_finished")

random.seed(0)
training_size=int(size*training_percentage)
training_label=list(np.sort(random.sample(range(size),training_size)))
allmatnum = [i for i in range(0, (O+1)*K)]
dis_sub=dis[np.ix_(allmatnum,training_label,training_label)]
family_sub=[family[training_label[j]] for j in range(0,training_size)]

torch.manual_seed(0)
weight=torch.rand((O+1)*K,requires_grad=True)
dis_sub_mean=np.zeros((O+1)*K)
for i in range(0,(O+1)*K):
    dis_sub_mean[i]=np.mean(dis_sub[i,:,:])
    dis[i,:,:]=dis[i,:,:]/dis_sub_mean[i]
    dis_sub[i,:,:]=dis_sub[i,:,:]/dis_sub_mean[i]
    dis_sub[i,:,:]=dis_sub[i,:,:]+10000*np.eye(training_size)
    weight.data[i]=weight.data[i]*dis_sub_mean[i]
weight.data=weight.data/torch.norm(weight.data, p=1, dim=0)
dis_sub_torch=torch.tensor(dis_sub).to(torch.float64)
dis_torch=torch.tensor(dis).to(torch.float64)

def f(mat,weight,M):
    disfinal=torch.matmul(weight.reshape(1,(O+1)*K),mat.reshape((O+1)*K,mat.size(1)*mat.size(2))).reshape(mat.size(1),mat.size(2))
    dis2=torch.nn.functional.normalize(1.0/disfinal,p=1,dim=0,eps=0)
    distem2=dis2
    for i in range(1,10):
        distem1=torch.pow(input=distem2,exponent=5)
        distem2=torch.nn.functional.normalize(distem1,p=1,dim=0,eps=0)
    dis3=torch.mul(distem2,M)
    return torch.sum(dis3)

def sumlist(fa):
    s=len(fa)
    M=np.zeros((s,s))
    for i in range(0,s):
        for j in range(0,s):
            if fa[i]==fa[j]:
                M[i,j]=1
    return M
match_sub=torch.tensor(sumlist(family_sub))

def KNN1(mat,fa,weight):
    s=len(fa)
    cor=0
    disfinal=(torch.matmul(weight.reshape(1,(O+1)*K),mat.reshape((O+1)*K,mat.size(1)*mat.size(2))).reshape(mat.size(1),-1)).numpy()
    disfinal=disfinal+np.diag([float("inf")] *s)
    for i in range(0,s):
        tem=disfinal[:,i]
        j=tem.argmin()
        if fa[i]==fa[j]:
            cor=cor+1
    return float(cor)/float(s)

def KNN2(mat,fa,weight):
    s=len(fa)
    trainnum=training_size
    testnum=s-training_size
    cor_train=0
    cor_test=0
    disfinal=(torch.matmul(weight.reshape(1,(O+1)*K),mat.reshape((O+1)*K,mat.size(1)*mat.size(2))).reshape(mat.size(1),-1)).numpy()
    disfinal=disfinal+np.diag([float("inf")] *s)
    for i in range(0,s):
        tem=disfinal[:,i]
        j=tem.argmin()
        if fa[i]==fa[j]:
            if i in training_label:
                cor_train+=1
            else:
                cor_test+=1
    return [float(cor_train)/float(trainnum),float(cor_test)/float(testnum),float(cor_train+cor_test)/float(s)]

result=[] #Sn0_score,S_score,Train_acc,Test_acc,Importance,dis_sub_mean
Sn0_score=[]
S_score=[]
Train_acc=[]
Test_acc=[]
Importance=[]
i=0
ep=-1
while i<P:
    ep+=1
    print("epoch:"+str(ep),"Patience:",str(i))
    y=f(dis_sub_torch,weight,match_sub)
    K1=KNN1(dis_sub_torch,family_sub,weight.data)
    K2=KNN2(dis_torch,family,weight.data)
    print("Sn0, S, and Test_Acc",y.data/float(training_size),K1, K2[1])
    Sn0_score.append(y.data/float(training_size))
    S_score.append(K1)
    Train_acc.append(K2[0])
    Test_acc.append(K2[1])
    weight_tem=weight.data
    Importance.append(weight_tem.numpy())
    y.backward(retain_graph=True)
    dwtem=lr*torch.norm(weight.data, p=1, dim=0)*torch.nn.functional.normalize(weight.grad.data,dim=0,p=1,eps=0)
    newweight=weight.data+dwtem
    newweight[newweight<0]=0
    x=f(dis_sub_torch,newweight,match_sub)
    while x.data-y.data<0:
        dwtem=dwtem/2
        newweight=weight.data+dwtem
        newweight[newweight<0]=0
        x=f(dis_sub_torch,newweight,match_sub)
    if x.data-y.data<0.0001*float(training_size):
        i+=1
    else:
        i=0
    weight.data=newweight
    weight.grad.data.zero_()
    sys.stdout.flush()

result.append(Sn0_score)
result.append(S_score)
result.append(Train_acc)
result.append(Test_acc)
result.append(Importance)
result.append(dis_sub_mean)
print(result)
