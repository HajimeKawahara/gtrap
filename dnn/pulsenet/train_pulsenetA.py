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
import iopulsenet as io
import glob
import re
import argparse
import os
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN classifier')
    parser.add_argument('-d', nargs=1, default=["/home/kawahara/gtrap/examples/train/train0/data"],help='directory containing train sets generated by gtls_mockkepler.', type=str)
    parser.add_argument('-o', nargs=1, default=["pulsenetA_train0"],help='directory containing train sets generated by gtls_mockkepler or gtls_slctess.', type=str)
    parser.add_argument('-e', nargs=1, default=[20],help='number of epochs', type=int)

    import datetime    
    args = parser.parse_args()

    output=args.o[0]+datetime.date.today().strftime("%Y_%m_%d")+".h5"
    
    traindir=args.d[0]
    flist=sorted(glob.glob(os.path.join(traindir,'*.npz')))
    print(len(flist))
    lab,X,Xw,info=io.makearr(flist)
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
    
    concat = concatenate([wide,local], axis=1)
    concat = Dense(units=512,activation=act)(concat)
    concat = Dense(units=512,activation=act)(concat)
    concat = Dense(units=512,activation=act)(concat)
    concat = Dense(units=512,activation=act)(concat)
    concat = Dense(units=1,activation="sigmoid")(concat)

    model = Model(inputs=[inwide,inlocal],outputs=concat)    
    model.compile(optimizer="adam",loss="binary_crossentropy")
    print(model.summary())
    model.fit([Xw,X],lab,epochs=args.e[0],validation_split=0.2)
    model.save(output)
