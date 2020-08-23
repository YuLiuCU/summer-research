# load packages
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, LSTM, Input, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB


import time
import random


# set random seeds
# np.random.seed(11)
# tf.random.set_seed(26)

# limit gpu usage for keras
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def data_classification(X,  T):
    [N, D] = X.shape
    df = np.array(X)

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,))


train_data_csv=pd.read_csv('train_data.csv')
test_data_csv=pd.read_csv('test_data.csv')

train_raw_csv=pd.read_csv('train_raw.csv')
test_raw_csv=pd.read_csv('test_raw.csv')

train_label_csv=pd.read_csv('train_label.csv')
test_label_csv=pd.read_csv('test_label.csv')

train_std_csv=pd.read_csv('train_std.csv')
test_std_csv=pd.read_csv('test_std.csv')

train_mean_csv=pd.read_csv('train_mean.csv')
test_mean_csv=pd.read_csv('test_mean.csv')

train_data = train_data_csv.to_numpy()
train_data = train_data[:,1:]
test_data = test_data_csv.to_numpy()
test_data = test_data[:,1:]

train_raw = train_raw_csv.to_numpy()
train_raw = train_raw[:,1:]
test_raw = test_raw_csv.to_numpy()
test_raw = test_raw[:,1:]

train_label = train_label_csv.to_numpy()
train_label = train_label[:,1:]
test_label = test_label_csv.to_numpy()
test_label = test_label[:,1:]

train_std = train_std_csv.to_numpy()
train_std = train_std[:,1:]
test_std = test_std_csv.to_numpy()
test_std = test_std[:,1:]

train_mean = train_mean_csv.to_numpy()
train_mean = train_mean[:,1:]
test_mean = test_mean_csv.to_numpy()
test_mean = test_mean[:,1:]

trainX_CNN = data_classification(train_data, T=100)
testX_CNN = data_classification(test_data, T=100)

trainY_CNN = to_categorical(train_label, 3)
testY_CNN = to_categorical(test_label, 3)


def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))
    
    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)
    
    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)

    # use the MC dropout here
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def gen_grad(raw,y,std,mean):
    R=np.reshape(raw,(1,100,40,1))
    S=np.reshape(std,(1,100,40,1))
    M=np.reshape(mean,(1,100,40,1))
    w=tf.constant(R)
    with tf.GradientTape() as tape:
        tape.watch(w)
        A=(w-M)/S
        loss=tf.keras.losses.categorical_crossentropy(y,deeplob(A))
    grad=tape.gradient(loss,w)
    grad=np.array(grad)
    grad=np.reshape(grad,(100,40))
    for i in range(20):
        grad[:,2*i]=0
    return grad

def proj(x,xn,budget):
    dx=xn-x
    a=np.array([x[:,2*i] for i in range(20)])
    a=np.reshape(a,(2000,1))
    b=np.array([dx[:,2*i+1] for i in range(20)])
    b=np.reshape(b,(2000,1))
    m=gp.Model()
    X=m.addMVar(2000)
    m.setObjective(X@np.eye(2000)@X-2*np.transpose(b)@X,GRB.MINIMIZE)
    m.addConstr(np.transpose(a)@X<=budget)
    m.setParam( 'OutputFlag', False )
    m.optimize()
    ans=X.X
    ans=np.reshape(ans,(100,20))
    XX=x
    for i in range(20):
        XX[:,2*i+1]+=ans[:,i]
    return XX

def pgd(x,y,std,mean,budget,eps,n):
    xn=x
    for i in range(n):
        dx=eps*gen_grad(xn,y,std,mean)
        xn=xn+dx
        xn=proj(x,xn,budget)
    for i in range(20):
        xn[:,2*i+1]=np.floor(xn[:,2*i+1])
    return xn

def compute_budget(raw):
    a=np.array([raw[:,2*i] for i in range(20)])
    b=np.array([raw[:,2*i+1] for i in range(20)])
    return np.sum(a*b)

deeplob = create_deeplob(100, 40, 64)
deeplob.load_weights('CP/cp_origin')
# deeplob.load_weights('CP2/cp_origin')

loss,acc = deeplob.evaluate(trainX_CNN, trainY_CNN, verbose=2)

itr=10

n=np.shape(trainY_CNN)
N=n[0]
n=640



trainX_CNN_ADV=np.zeros((n,100,40,1))
time_start=time.time()
for j in range(itr):
    print(time.time()-time_start)
    T=0
    generator=random.sample(range(1,N),n)
    trainY_CNN_ADV=trainY_CNN[generator,:]
    for i in generator:
        X=train_raw[i:i+100,:]
        Y=trainY_CNN[i,:]
        S=train_std[i:i+100,:]
        M=train_mean[i:i+100,:]
        # budget=1e-2*compute_budget(X)
        Xn=pgd(X,Y,S,M,1e4,1e4,20)
        Xn=(Xn-M)/S
        Xn=np.reshape(Xn,(100,40,1))
        trainX_CNN_ADV[T,:,:,:]=Xn
        T=T+1
    for i in range(1):
        deeplob.fit(trainX_CNN_ADV, trainY_CNN_ADV, epochs=10, batch_size=64, verbose=2)
        deeplob.save_weights('CP3/cp_origin')




n=np.shape(testY_CNN)
N=n[0]
n=1000
generator=random.sample(range(1,N),n)

testX_CNN_ADV_ori=testX_CNN[generator,:,:,:]
testY_CNN_ADV=testY_CNN[generator,:]
deeplob.load_weights('CP/cp_origin')
loss1,acc1 = deeplob.evaluate(testX_CNN_ADV_ori, testY_CNN_ADV, verbose=2)

testX_CNN_ADV=np.zeros((n,100,40,1))
T=0
for i in generator:
    X=test_raw[i:i+100,:]
    Y=testY_CNN_ADV[T,:]
    S=test_std[i:i+100,:]
    M=test_mean[i:i+100,:]
    # budget=1e-2*compute_budget(X)
    Xn=pgd(X,Y,S,M,1e5,1e5,5)
    Xn=(Xn-M)/S
    Xn=np.reshape(Xn,(100,40,1))
    testX_CNN_ADV[T,:,:,:]=Xn
    T=T+1

loss2,acc2 = deeplob.evaluate(testX_CNN_ADV, testY_CNN_ADV, verbose=2)
deeplob.load_weights('CP3/cp_origin')
loss3,acc3 = deeplob.evaluate(testX_CNN_ADV_ori, testY_CNN_ADV, verbose=2)
loss4,acc4 = deeplob.evaluate(testX_CNN_ADV, testY_CNN_ADV, verbose=2)


