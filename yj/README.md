# Information

## List of contents

* README.md - Summary of this project
* fer2013.py - Emotion recognition using fer2013 dataset

## Summary of Keras documentation 

### Keras model 저장
* model.save(filepath): 모델 저장하기. 이 때 저장할 정보는 다음과 같습니다.
    * 모델의 구조(architecture, 아키텍처)
    * 모델이 가진 weight
    * 학습설정(configuration)
    * 최적화 상태, 학습중지상태에서 학습재개

* load_model(filepath): 저장한 모델 불러오기`

### 유용한 option
* EarlyStopping: validation loss가 더 이상 감소하지 않을 때 학습중지옵션
```buildoutcfg
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, cllbacks=[early_stopping])
```

## Reference
* keras.io: keras documentation