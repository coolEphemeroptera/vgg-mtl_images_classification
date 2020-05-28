from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from keras.callbacks import Callback,EarlyStopping
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
import utils
import os

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# base 模块
def bn(x):
    return BatchNormalization(axis=-1)(x)

def conv2d(x,n_fmaps,k=3):
    return Conv2D(n_fmaps, (k, k), strides=(1,1), activation='relu', padding='same')(x)

def maxpool(x):
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

def dense(x,n_units,activation="relu"):
    x = Dropout(0.25)(x)
    return Dense(n_units,activation=activation)(x)

def softmax(x,n_classes,name):
    x = Dropout(0.25)(x)
    return Dense(n_classes,activation='softmax',name=name)(x)

# vgg_16
def vgg_16(inputs, n_classes, scale=1):
    # cnn-1
    x = bn(conv2d(inputs, int(64 * scale)))
    x = bn(conv2d(x, int(64 * scale)))
    x = maxpool(x)

    # cnn-2
    x = bn(conv2d(x, int(128 * scale)))
    x = bn(conv2d(x, int(128 * scale)))
    x = maxpool(x)

    # cnn-3
    x = bn(conv2d(x, int(256 * scale)))
    x = bn(conv2d(x, int(256 * scale)))
    x = bn(conv2d(x, int(256 * scale), k=1))
    x = maxpool(x)

    # cnn-4
    x = bn(conv2d(x, int(512 * scale)))
    x = bn(conv2d(x, int(512 * scale)))
    x = bn(conv2d(x, int(512 * scale), k=1))
    x = maxpool(x)

    # cnn-5
    x = bn(conv2d(x, int(512 * scale)))
    x = bn(conv2d(x, int(512 * scale)))
    x = bn(conv2d(x, int(512 * scale), k=1))
    x = maxpool(x)

    # fc-6,7,8
    x = Reshape([7 * 7 * int(512 * scale)])(x)
    x = dense(x, int(4096 * scale))
    x = dense(x, int(4096 * scale))
    x = dense(x, int(1000 * scale))

    # softmax
    x = softmax(x,n_classes)
    return x


# vgg_11
def vgg_11(inputs, n_classes):
    # cnn-1
    x = bn(conv2d(inputs, 32))
    x = maxpool(x)

    # cnn-2
    x = bn(conv2d(x, 64))
    x = maxpool(x)

    # cnn-3
    x = bn(conv2d(x, 128))
    x = bn(conv2d(x, 128))
    x = maxpool(x)

    # cnn-4
    x = bn(conv2d(x, 256))
    x = bn(conv2d(x, 256))
    x = maxpool(x)

    # cnn-4
    x = bn(conv2d(x, 256))
    x = bn(conv2d(x, 256))
    x = maxpool(x)

    # fc-6,7,8
    x = Reshape([7 * 7 * 256])(x)
    x = dense(x, 1024)
    x = dense(x, 1024)
    x = dense(x, 256)

    # softmax
    x = softmax(x, n_classes)
    return x 

# 提前终止
es = EarlyStopping(verbose=1,patience=3,restore_best_weights=True)

if __name__ == "__main__":
    # 超参数
    IMG_SIZE = (224, 224)
    N_CLASSES = 105
    BATCH_SIZE = 64
    EPOCHS = 20 
    train_file = "./data.train"
    dev_file = "./data.dev"
    test_file = "./data.test"

    # 数据准备
    train_lst = utils.reading(train_file)
    dev_lst = utils.reading(dev_file)
    test_lst = utils.reading(test_file)
    n_batchs = len(train_lst)//BATCH_SIZE

    dev_data,dev_labels = utils.loading(dev_lst,n_classes=N_CLASSES)
    test_data, test_labels = utils.loading(test_lst, n_classes=N_CLASSES)
    train_generator = utils.data_generator(train_lst,batch_size=BATCH_SIZE,n_classes=N_CLASSES)

    # 模型 输入
    inputs = Input(name='the_inputs', shape=(IMG_SIZE[0], IMG_SIZE[1],3), dtype='float32')

    # 定义模型
    model = Model(inputs=inputs, outputs=vgg_16(inputs, N_CLASSES, 0.25))
    model.summary()
    # opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001, epsilon=10e-8) 
    opt = SGD(lr=0.01, decay=0.001, momentum=0.9, nesterov=True)

    # 单GPU训练
    # model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    # model.fit_generator(generator=train_generator,steps_per_epoch=n_batchs,epochs=EPOCHS,
    #                     callbacks=[es],validation_data=(dev_data,dev_labels),validation_steps=n_batchs) 

    # 多gpu训练
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    parallel_model.fit_generator(generator=train_generator,steps_per_epoch=n_batchs,epochs=EPOCHS,
                        callbacks=[es],validation_data=(dev_data,dev_labels),validation_steps=n_batchs)


