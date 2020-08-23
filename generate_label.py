import pandas as pd
import pickle
import numpy as np

def avg_prize(n):
    sum_p = 0
    sum_s = 0
    for i in range(20):
        p = n[i*2]
        s = n[i*2+1]
        sum_p += p*s
        sum_s += s
    return sum_p/sum_s

def gen_label(T,k,alpha,X):
    x = len(X)
    label = np.ones((x,1))
    for i in range(x-k):
        m_plus = np.mean(X[i+1:i+k+1])
        l = (m_plus - X[i])/X[i]
        if l > alpha:
            label[i] = 2
        elif l < -alpha:
            label[i] = 0
    return label[T-1:]

T = 100
k = 3
alpha = 0.000015

train_data_csv=pd.read_csv('train_raw.csv')
train_data = train_data_csv.to_numpy()
train_data = train_data[:,1:]
[a,b] = np.shape(train_data)
prize = np.zeros((a,1))
for i in range(a):
    prize[i] = avg_prize(train_data[i,:])
label_train = gen_label(T,k,alpha,prize)
[AA,BB]=np.unique(label_train,return_counts=True)




val_data_csv=pd.read_csv('val_raw.csv')
val_data = val_data_csv.to_numpy()
val_data = val_data[:,1:]
[a,b] = np.shape(val_data)
prize = np.zeros((a,1))
for i in range(a):
    prize[i] = avg_prize(val_data[i,:])
label_val = gen_label(T,k,alpha,prize)
# [AA,BB]=np.unique(label_val,return_counts=True)


test_data_csv=pd.read_csv('test_raw.csv')
test_data = test_data_csv.to_numpy()
test_data = test_data[:,1:]
[a,b] = np.shape(test_data)
prize = np.zeros((a,1))
for i in range(a):
    prize[i] = avg_prize(test_data[i,:])
label_test = gen_label(T,k,alpha,prize)
# [AA,BB]=np.unique(label_test,return_counts=True)

pd.DataFrame(label_train).to_csv('train_label.csv')
pd.DataFrame(label_val).to_csv('val_label.csv')
pd.DataFrame(label_test).to_csv('test_label.csv')