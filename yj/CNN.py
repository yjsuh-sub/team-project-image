#CNN

from keras.utils import np_utils
from time import time
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from keras import optimizers

path = 'C:/Anaconda2/bin/fer2013/fer2013' # 이미지 파일 경로
df = pd.read_csv('%s/all.csv'%path) # 정제한 이미지 파일
df.drop('Unnamed: 0', axis=1, inplace=True) # 쓸데없는 컬럼 drop
df_train = df[df.Usage == 'Training'] # Training이라고 라벨되어있는 항목만 가져오기
X_train = np.array(df_train.iloc[:, 2:]).astype('float32') / 255. # float 타입 변경, scaling
X_train = X_train.reshape(28709, 48, 48, 1) # 입력형태에 맞게 reshape
y_train = np.array(df_train.iloc[:, 0]) # y는 1차원이고 카테고리값이라 scaling 필요없음
df_test = df[df.Usage == 'PrivateTest'] # 교차검증용 테스트셋
X_test = np.array(df_test.iloc[:, 2:]).astype('float32') / 255.
X_test = X_test.reshape(3589, 48, 48, 1)
y_test = np.array(df_test.iloc[:, 0])

Y_train = np_utils.to_categorical(y_train, 7) # 원핫인코딩
Y_test = np_utils.to_categorical(y_test, 7)

# 행렬 모양 살피기
#print(X_train.shape, X_train.dtype)
#print(y_train.shape, y_train.dtype, y_train[:3])
#print(X_test.shape, X_test.dtype)
#print(y_test.shape, y_test.dtype)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2

np.random.seed(0)

model = Sequential()

# strong overfitting >> regularizer 수 약간 늘리거나 dropout 비율 늘리기
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1), padding='same', kernel_regularizer=l2(0.001)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

#불필요해보이는 레이어 비활성화ㄴ
#model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.0002)))
#model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.0002)))
#model.add(MaxPooling2D())
#model.add(Dropout(0.4))

#model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.0002)))
#model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.0002)))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

opt = optimizers.Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 실행코드 + 시간측정
ct = time()
hist = model.fit(X_train, Y_train, epochs=40, batch_size=100, validation_data=(X_test, Y_test), verbose=2)
print('Wall time:', time() - ct)
# 생성한 모델 저장
model.save('fer2013.hdf5')
# loss graph와 training 정확도+test정확도 그래프
plt.figure(figsize=(8, 15))
plt.subplot(211)
plt.plot(hist.history['loss'], label='loss')
plt.title('loss')
plt.legend()
plt.subplot(212)
plt.title('accuracy')
plt.plot(hist.history["acc"], label="training accuracy")
plt.plot(hist.history["val_acc"], label='test accuracy')
plt.legend()
plt.tight_layout()
plt.show()