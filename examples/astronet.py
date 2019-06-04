import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras import backend
from keras.layers.merge import concatenate

import glob
import re
import argparse
import os

def makearr(flist):
    lab=[]
    X=[]
    Xw=[]
    info=[]
    for fn in flist:
        d=np.load(fn,allow_pickle=True)
        lab.append(d["arr_0"][0])
        Xtmp=d["arr_1"]
        Xtmp[Xtmp!=Xtmp]=0.0
        X.append(Xtmp+1.0)

        Xwtmp=d["arr_2"]
        Xwtmp[Xwtmp!=Xwtmp]=0.0        
        Xw.append(Xwtmp+1.0)
        
        info.append(d["arr_3"])
    lab=np.array(lab).astype(np.int32)
    X=np.array(X).reshape(np.shape(X)[0],np.shape(X)[1],1)
    Xw=np.array(Xw).reshape(np.shape(Xw)[0],np.shape(Xw)[1],1)
    info=np.array(info)
    
    return lab,X,Xw,info

#precision
def Precision(y_true, y_pred):
    true_positives = backend.sum(backend.cast(backend.greater(backend.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = backend.sum(backend.cast(backend.greater(backend.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + backend.epsilon())
    return precision

#recall
def Recall(y_true, y_pred):
    true_positives = backend.sum(backend.cast(backend.greater(backend.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = backend.sum(backend.cast(backend.greater(backend.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + backend.epsilon())
    return recall

#f-measure
def Fvalue(y_true, y_pred):
    p_val = Precision(y_true, y_pred)
    r_val = Recall(y_true, y_pred)
    f_val = 2*p_val*r_val / (p_val + r_val)

    return f_val



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN classifier')
    parser.add_argument('-d', nargs=1, default=["/home/kawahara/gtrap/examples/mocklc_clean"],help='directory containing train sets generated by gtls_mockkepler.', type=str)
    parser.add_argument('-o', nargs=1, default=["astronet"],help='directory containing train sets generated by gtls_mockkepler.', type=str)
    parser.add_argument('-e', nargs=1, default=[20],help='number of epochs', type=int)


    import datetime
    
    args = parser.parse_args()

    output=args.o[0]+datetime.date.today().strftime("%Y_%m_%d")+".h5"
    
    traindir=args.d[0]
    flist=sorted(glob.glob(os.path.join(traindir,'*.npz')))
    print(len(flist))
    lab,X,Xw,info=makearr(flist)
    
    inlocal = Input(shape=(np.shape(X)[1],np.shape(X)[2]))
    inwide = Input(shape=(np.shape(Xw)[1],np.shape(Xw)[2]))

    pad="same"
    act="relu"
    strmp=2
    
    wide = Conv1D(16,5,activation=act,padding=pad)(inwide)
    wide = Conv1D(16,5,activation=act,padding=pad)(wide)
    wide = MaxPooling1D(5,strides=strmp)(wide)
    wide = Conv1D(32,5,activation=act,padding=pad)(wide)
    wide = Conv1D(32,5,activation=act,padding=pad)(wide)
    wide = MaxPooling1D(5,strides=strmp)(wide)
    wide = Conv1D(64,5,activation=act,padding=pad)(wide)
    wide = Conv1D(64,5,activation=act,padding=pad)(wide)
    wide = MaxPooling1D(5,strides=strmp)(wide)
    wide = Conv1D(128,5,activation=act,padding=pad)(wide)
    wide = Conv1D(128,5,activation=act,padding=pad)(wide)
    wide = MaxPooling1D(5,strides=strmp)(wide)
    wide = Conv1D(256,5,activation=act,padding=pad)(wide)
    wide = Conv1D(256,5,activation=act,padding=pad)(wide)
    wide = MaxPooling1D(5,strides=strmp)(wide)
    wide = Flatten()(wide)

    local = Conv1D(16,5,activation=act,padding=pad)(inlocal)
    local = Conv1D(16,5,activation=act,padding=pad)(local)
    local = MaxPooling1D(7,strides=strmp)(local)
    local = Conv1D(32,5,activation=act,padding=pad)(local)
    local = Conv1D(32,5,activation=act,padding=pad)(local)
    local = MaxPooling1D(7,strides=strmp)(local)
    local = Flatten()(local)

#    print(np.shape(X)[1],np.shape(X)[2])
#    print(local)
#    print("-----------")
#    print(np.shape(Xw)[1],np.shape(Xw)[2])
#    print(wide)
    
    concat = concatenate([wide,local], axis=1)
    concat = Dense(units=512,activation=act)(concat)
    concat = Dense(units=512,activation=act)(concat)
    concat = Dense(units=512,activation=act)(concat)
    concat = Dense(units=512,activation=act)(concat)
    concat = Dense(units=1,activation="sigmoid")(concat)

    model = Model(inputs=[inwide,inlocal],outputs=concat)    
    model.compile(optimizer="adam",loss="binary_crossentropy")

    print(model.summary())
    
#    useF=300
#    model.fit([Xw[:useF,:,:],X[:useF,:,:]],lab[:useF],epochs=5,validation_data=([Xw[useF:,:,:],X[useF:,:,:]],lab[useF:]))
    model.fit([Xw,X],lab,epochs=args.e[0],validation_split=0.2)
    model.save(output)
