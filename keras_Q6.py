import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

BATCH_SIZE = 128  # batch大小，每处理128个样本进行一次梯度更新
NUM_CLASSES = 10  # 类别数
NUM_EPOCHS = 10   # 迭代次数

# download and load the data (split them between train and test sets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# expand the channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# 数据转为float32型并归一化
# make the value of pixels from [0, 255] to [0, 1] for further process
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrics
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# define the model
model = Sequential()
# 第一层为二维卷积层
# 32 为filters卷积核的数目，也为输出的维度
# kernel_size 卷积核的大小，3x3
# 激活函数选为 relu
# 第一层必须包含输入数据规模input_shape这一参数
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 再加一层卷积，64个卷积核
model.add(Conv2D(64, (3, 3), activation='relu'))
# 加最大值池化
model.add(MaxPooling2D(pool_size=(2, 2)))
# 加Dropout，断开神经元比例为25%
model.add(Dropout(0.25))
# 加Flatten，数据一维化
model.add(Flatten())
# 加Dense，输出128维
model.add(Dense(128, activation='relu'))
# 再一次Dropout
model.add(Dropout(0.5))
# 最后一层为Softmax，输出为10个分类的概率
model.add(Dense(NUM_CLASSES, activation='softmax'))

# define the object function, optimizer and metrics
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# train
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test))

# evaluate
score_train = model.evaluate(x_train, y_train, verbose=0)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0]*100, score_train[1]*100))
score_test = model.evaluate(x_test, y_test, verbose=0)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0]*100, score_test[1]*100))