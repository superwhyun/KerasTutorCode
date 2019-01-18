
# MNIST는 10종류(클래스 수)에 대해 각 1000장씩 총 60000장의 사진을 가지고 있음
# epochs 수 x 60000 개의 input을 가지고 놈.
# 실행하면, 아래와 같은 로그가 나올거임
#   35712/60000 [================>.............] - ETA: 2:24 - loss: 0.1544 - acc: 0.9519

#
# 주의사항: tf.keras~~ 와 keras.~~~ 는 다른거다. 두개를 혼용해서 사용하면 좇된다. 에러 바바바박 뜨고 막 그란다. 
#

# 
# TODO
#   [v] 학습된 weight 저장하기/불러오기
#   - 학습된 weight에 추가로 학습하기 
#   - 클래스 종류 추가하기? ==> 가능할까? 아마 안될끄이다...
#
#   [v] Tensorboard로 보여주기
#        - https://tykimos.github.io/2017/07/09/Training_Monitoring/
#        - https://tykimos.github.io/2017/08/07/Keras_Install_on_Mac/
#   - Serving 모드로 image를 입력받아 검색결과 보여주기
#

import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import adam
import keras

class MNIST_KERAS:
    def __init__(self, batch_size, num_class, epochs, imgHeight, imgWidth, lossFunc='categorical_crossentropy', optiFunc='adam', metrics=['accuracy']):
        self.batch_size = batch_size
        self.number_of_class = num_class
        self.epochs = epochs
        self.img_height = imgHeight
        self.img_width = imgWidth

        self.loss = lossFunc
        self.optimizer = optiFunc
        self.metrics = metrics


    def prepare_data(self):

        # download MNIST data sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_height, self.img_width, 1)
        self.x_test  = self.x_test.reshape(self.x_test.shape[0], self.img_height, self.img_width, 1)
        

        self.input_shape = (self.img_height, self.img_width, 1)

        # Type conversion
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')


        # Normalization
        self.x_train /= 255
        self.x_test /= 255

        # One hot encoding으로 각 클래스마다 하나의 output으로 연결되도록 함
        self.y_train = tf.keras.utils.to_categorical(y=self.y_train, num_classes=self.number_of_class)
        self.y_test = tf.keras.utils.to_categorical(y=self.y_test, num_classes=self.number_of_class)


    def save_model(self, model_file, weight_file):
    
        model_json = self.model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(weight_file)
        print("Saved model to disk")        

    def load_model(self, model_file, weight_file):

        try:
            json_fd = open(model_file)
            weight_fd = open(weight_file)
        except:
            return False

        loaded_model_json=json_fd.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(weight_file)
        
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        print("Loading from disk done")        
        return True
        

    def create_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = (3,3),
            activation='relu',
            input_shape = self.input_shape
        ))
        self.model.add(tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size = (3,3),
            activation='relu'
        ))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Dropout(rate=0.3))

        self.model.add(tf.keras.layers.Flatten())                                            # fully connected layer로 가려면, 2차원 data를 1차원으로 바꿔줘야 하며, flatten layer가 그 역할을 함
        self.model.add(tf.keras.layers.Dense(units=1024, activation='relu'))                 # fully connected layer로 만들어 줌
        self.model.add(tf.keras.layers.Dropout(rate=0.3))
        self.model.add(tf.keras.layers.Dense(units=self.number_of_class, activation='softmax'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)


    def train(self):

        tb_hist = tf.keras.callbacks.TensorBoard(
            log_dir='./graph', 
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )
     
        self.model.fit(
            x=self.x_train, 
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_test, self.y_test),
            callbacks=[tb_hist]   
        )
    
    def test(self):

        print('Testing for train set x ', len(self.x_train), ' EA')

        train_loss, train_accuray = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print("Train data loss:", train_loss)
        print("Train data accuracy:", train_accuray)

        print('Testing for test set x ', len(self.x_test), ' EA')
        test_loss, test_accuray = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Test data loss:", test_loss)
        print("Test data accuracy:", test_accuray)


mnistKeras = MNIST_KERAS(
                batch_size=128,
                num_class=10,
                epochs=2,
                imgHeight=28,
                imgWidth=28
            )

mnistKeras.prepare_data()

if(mnistKeras.load_model('model.json', 'model.h5')==False):
    mnistKeras.create_model()
    mnistKeras.train()
    mnistKeras.save_model('model.json', 'model.h5')    

mnistKeras.test()


