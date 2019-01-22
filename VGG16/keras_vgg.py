from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np


# decode_predictions을 해서 결과를 편안히 보려면, include_top을 True로 해야 함
#   - 'True'의 의미는 final dense layer를 포함할지 말지를 결정하는 것
#       - 이 final layer가 class를 결정해주는 파트를 포함시키게 해서 주로 그냥 predict만 하고자 할때 사용
#   - 변경하고 실행하면, weight file을 다시 다운 받음 (553MB)
# False일 경우
#   - 네트워크 형태나 학습된 weight 값은 따다 쓰지만, parameter나 class등을 내가 임의 조정하고 싶을 때
#   - 즉, 입력 파라미터(아마 이미지 크기, 비율 등)와 출력되는 클래스(이름 등이겠지)를 변경할 수 있음
#   - predict한 결과로 512 feature map with 7x7 size만을 리턴함.
#       - (1, 7, 7, 512)
model = VGG16(weights='imagenet', include_top=True)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prediction = model.predict(x)
label = decode_predictions(prediction)


# print('Predicted:', decode_predictions(prediction, top=3)[0])

for (i, (imagenetID, label, prob)) in enumerate(label[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
