import numpy as np
import csv
import sys
from Bio import SeqIO

def distance(X):
  n,m = X.shape
  G = np.dot(X,X.T)
  H = np.tile(np.diag(G), (n,1))
  return np.sqrt(H + H.T - 2*G)

def Kmer(sequence, K,o):
    m=4**K
    na_vect=[0]*((o+1)*m)
    pos_sum=[0]*m
    n=len(sequence)-(K-1)
    index_map = {  'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3  }
    for i in range(0, n):
        flag=1
        for l in range(0,K):
            if sequence[i+l] not in index_map.keys():
                flag=0
        if flag == 0:
            continue
        tem=index_map[sequence[i]]
        for l in range(1,K):
            tem=4*tem+index_map[sequence[i+l]]
        na_vect[tem] += 1
        pos_sum[tem] += i+1
    for k in range(0,m):
        if na_vect[k] != 0:
            na_vect[k+m] = pos_sum[k] / na_vect[k]
        else:
            na_vect[k+m]=0
    for t in range(2, o+1):
        m_sum = [0] * m
        for i in range(0, n):
            flag=1
            for l in range(0,K):
                if sequence[i+l] not in index_map.keys():
                    flag=0
            if flag == 0:
                continue
            tem=index_map[sequence[i]]
            for l in range(1,K):
                tem=4*tem+index_map[sequence[i+l]]
            m_sum[tem] += ( i + 1 - na_vect[tem+m] ) ** t
        for k in range(0,m):
            if na_vect[k] != 0:
                na_vect[k+t*m] = m_sum[k] / n ** (t-1) / na_vect[k] ** (t-1)
            else:
                na_vect[k+t*m]=0
    return na_vect


seqnum=11559
KK=9
O=2
for K in range(1,KK+1):
    nv=np.zeros((seqnum, (O+1)*(4**K)))
    i=0
    for read in SeqIO.parse("clean.fasta", "fasta"):
        s=(read.seq).__str__()
        nvtem=Kmer(s,K,O)
        nv[i,:]=nvtem
        i+=1
        print("readnv:",K,i)
        sys.stdout.flush()
    for j in range(0,O+1):
        nv_Kj=nv[:,j*(4**K):(j+1)*(4**K)]
        mat_Kj=distance(nv_Kj)
        outname='Order_dis'+str(K)+'_'+str(j)+'.csv'
        np.savetxt(outname,mat_Kj,delimiter=',')
    print("calculation:",K)
    sys.stdout.flush()




