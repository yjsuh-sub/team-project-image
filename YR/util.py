import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def print_img(row, col, img_target, idx_target, idx):
    fig = plt.figure(figsize=(col*2.5, row*2.5))
    
    for i in range(len(idx)):
        plt.subplot(row, col, (i+1))
        plt.imshow(img_target[idx[i]].reshape((48, 48)), cmap='gray')
        plt.grid(False); plt.xticks([]); plt.yticks([]);
        lb = idx_target[idx[i]]
        
        if lb[0] == 1:
            plt.xlabel('0.Angry')
        elif lb[1] == 1:
            plt.xlabel('1.Disgust')
        elif lb[2] == 1:
            plt.xlabel('2.Fear')
        elif lb[3] == 1:
            plt.xlabel('3.Happy')
        elif lb[4] == 1:
            plt.xlabel('4.Sad')
        elif lb[5] == 1:
            plt.xlabel('5.Surprise')
        elif lb[6] == 1:
            plt.xlabel('6.Neutral')
    
    plt.show()

def check_generator(X, Y, arg):
    datagen = arg
    datagen.fit(X)
    
    # configure batch size and retrieve one batch of images
    for X_batch, Y_batch in datagen.flow(X, Y, batch_size=8, shuffle=False):
        break
    print_img(row=1, col=8, img_target=X_batch, idx_target=Y_batch, idx=[0, 1, 2, 3, 4, 5, 6, 7])

def print_ppscore(df, norm):
    dfRes0 = pd.DataFrame(df.pred0, columns=['pred0'])
    dfRes0['origin'] = norm
    dfRes0['result'] = df.pred0 == norm

    dfRes1 = pd.DataFrame(df.pred1, columns=['pred1'])
    dfRes1['origin'] = norm
    dfRes1['result'] = df.pred1 == norm

    acc0 = float((dfRes0['result'] == True).sum()) / len(dfRes0['result'])
    acc1 = float((dfRes1['result'] == True).sum()) / len(dfRes1['result'])
    print('acc0: {}'.format(acc0))
    print('acc1: {}'.format(acc1))