# Information

![alt text](https://github.com/yjsuh-sub/team-project-image/blob/master/yj/plutchik-emotions-chart.gif)

## List of contents

* README.md - Summary of this project
* fer2013.py - Emotion recognition using fer2013 dataset

## Research on Facial Expression Recognition(FER)
 - Method 1: [7] Y. Tang, "Deep Learning using Linear Support Vector Machines", 2013
 - Method 2: [8] Z. Yu and C. Zhang, “Image based static facial expression recognition with multiple deep network learning,” in ACM International Conference on Multimodal Interaction (MMI), 2015
 - Method 3: [9] B.-K. Kim, S.-Y. Dong, J. Roh, G. Kim, and S.-Y. Lee, “Fusing Aligned and Non-Aligned Face Information for Automatic Affect Recognition in the Wild: A Deep Learning Approach", 2016
 - Method 4: [11] B.-K. Kim, J. Roh, S.-Y. Dong, and S.-Y. Lee, "Hierarchical committee of deep convolutional neural networks for robust facial expression recognition", 2016, 
 - Method 5: [12] A. Mollahosseini, D. Chan, and M. H. Mahoor, “Going Deeper in Facial Expression Recognition using Deep Neural Networks", 2015
 - Method 6: [14] Z. Zhang, P. Luo, C.-C. Loy, and X. Tang, “Learning Social Relation Traits from Face Images", 2015
 - ref*: [15] X. Xiong and F. Torre, “Supervised descent method and its applications to face alignment", 2013

<h4><center> TABLE I: Preprocessing operations</center></h4>

|Method | FD | LM | Registration | Illumionation | Training cross validation(Public) |
|-------|----|----|-------------|----------------|---------------------------|
|Method 1| no| no|no | normalize | 71.2%(DLSVM L2)|  
|Method 2| several | no | no | histeq, LPF |72%(Estimate)|  
|Method 3 | no |ref* | rigid (LM) | several | 73.5%(Estimate) |  
|Method 4| several | ref* |rigid (LM) | several |72.5%(Estimate)|  
|Method 5| no | ref* | affine (LM) | no | 66.5%(Estimate) |  
|Method 6| no | ref* | indirect | no | 75%(Estimate) |  

 * FD: Facial Detection
 * LM: Facial Landmark Extraction
 * HISTEQ: Histogram Equalization
 * LPF: Linear Plane Fitting
 
Note that, The human accuracy on this dataset is around $65\pm5$% [1]


<h4><center> TABLE II: CNN Architectures</center></h4>

|Method | Architecture | Depth | Parameters |
|-------|----|----|-------------|
|Method 1| CPCPFF| 4 | 12.0 m | 
|Method 2| PCCPCCPCFFF | 8 | 6.2 m |
|Method 3 | CPCPCPFF | 5 | 2.4 m | 
|Method 4| CPCPCPFF | 5 | 4.8 m | 
|Method 5| CPCPIIPIPFFF | 11 | 7.3 m |
|Method 6| CPNCPNCPCFF | 6 | 21.3 m |

 * C: Convolutional
 * Pooling
 * Response-Normalization
 * I: Inception
 * F: Fully connected layers
 
<h4><center> TABLE III: Differences in terms of CNN training and inference </center></h4>

|Method | AD | AF | + Train | + Test | Ensemble |
|-------|:----:|:----:|:----:|--------|---------|
|Method 1| no | no | S, M | -| average |
|Method 2| no | no | A, M | A| weighted |
|Method 3 | no | yes | T, M, REG | ten-crop, REG| average |
|Method 4| no | no | T, M | - | hierarchy |
|Method 5| yes | no | ten-crop |-| -|
|Method 6| yes | yes | - |- |- |

* AD: Additional Training Data
* AF: Additional Features
* +: Data augmentation
* S: Similarity Transformation
* A: Affine Transformation
* T:Translation
* M: Horizontal Mirroring
* REG: Face Registration


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
* ModelCheckpoint: 각 epoch마다 모델 저장, filepath에는 epoch, logs의 key 등 사용 가능. 예) weights.{epoch:02d}-val_loss:.2f}.hdf5
```buildoutcfg
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1
```
`keys of log`
* on_epoch_end: acc, loss, val_loss, val_acc
* on_batch_begin: size - 현재 batch의 샘플 수
* on_batch_end: loss, acc

### Available loss functions
* mean_squared_error
* mean_absolute_error
* mean_absolute_percentage_error
* mean_squared_logarithmic_error
* squared_hinge
* hinge
* categorical_hinge
* logcosh
* categorical_crossentropy
* sparse_categorical_crossentropy
* binary_crossentropy
* kullback_leibler_divergence
* poisson
* cosine_proximity

## Reference
* keras.io: keras documentation
* [1] I. J. Goodfellow, D. Erhan, P. L. Carrier, A. Courville, M. Mirza, B. Hamner, W. Cukierski, Y. Tang, D. Thaler, D.-H. Lee, Y. Zhou,C. Ramaiah, F. Feng, R. Li, X. Wang, D. Athanasakis, J. Shawe-Taylor, M. Milakov, J. Park, R. Ionescu, M. Popescu, C. Grozea, J. Bergstra, J. Xie, L. Romaszko, B. Xu, Z. Chuang, and Y. Bengio, “Challenges in representation learning: A report on three machine learning contests,”Neural Networks, vol. 64, pp. 59–63, 2015.