# KerasTutorCode
Tutorial codes for Keras


## MNIST
- MNIST는 10종류(클래스 수)에 대해 각 1000장씩 총 60000장의 사진을 가지고 있음
- epochs 수 x 60000 개의 input을 가지고 놈.

### TODO

- ~~학습된 weight 저장하기/불러오기~~
- 학습된 weight에 추가로 학습하기 
- 클래스 종류 추가하기? ==> 가능할까? 아마 안될끄이다...
- ~~Tensorboard로 보여주기~~

- Serving 모드로 image를 입력받아 검색결과 보여주기

### References
  - https://tykimos.github.io/2017/07/09/Training_Monitoring/
  - https://tykimos.github.io/2017/08/07/Keras_Install_on_Mac/
  
## CIFAR
- CIFAR-10 : 10개의 클래스, 32x32 픽셀 이미지, 5만개 훈련 이미지, 1만개 테스트 이미지로 구성, 클래스당 6000개 이미지
- 작은 이미지에 하나의 객체만 포함됨
- CIFAR-100 : 100개의 클래스, 32x32 픽셀 이미지, 5만개 훈련 이미지, 1만개 테스트 이미지로 구성, 클래스당 600개 이미지



## Fashion-MNIST
- MNIST가 너무 쉬워서 대체용으로 만들어진 것. 구성은 MNIST와 유사


