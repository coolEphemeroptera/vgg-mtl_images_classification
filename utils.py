from keras.applications.vgg16 import  preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical
import numpy as np
import re
from random import shuffle

# linux to windows 路径转换
def path_lin2win(path):
    pattern = "/[a-z]/"
    position = re.findall(pattern,path)[0][1].upper()
    return re.sub(pattern,"%s:/"%position,path)


# 读取文件
def reading(path):
    file = open(path, mode='rt', encoding='UTF-8')
    lst = [line.replace('\n', '') for line in file.readlines()]
    file.close()
    return lst

# 读取图片
def read_img(path,SIZE=(224,224)):
    image = load_img(path, target_size=SIZE)
    image_data = img_to_array(image)/255
    image_data = image_data*2-1
    # image_data = preprocess_input(image_data)
    return image_data


# 标签 onehot
def onehot(labels,n_classes):
    return to_categorical(labels,num_classes=n_classes)


# 加载数据
def loading(lst,n_classes):
    data = []
    labels = []
    for n,line in enumerate(lst):
        img_path, label = line.split()
        # img_path = path_lin2win(img_path)
        label = onehot(int(label),n_classes)
        data.append(read_img(img_path))
        labels.append(label)
        # print("正在加载第 %06d 张图片：%s"%(n,img_path))
    return {"the_inputs": np.float32(np.asarray(data))},{"the_labels":np.int32(np.asarray(labels))}


# 构造生成器
def data_generator(lst,batch_size,n_classes):
    n_batchs = len(lst) // batch_size
    while True:
        shuffle(lst)
        for i in range(n_batchs):
            begin = i * batch_size
            end = begin + batch_size
            subs = lst[begin:end]
            data,labels = loading(subs,n_classes)
            yield data,labels

# test
if __name__ == "__main__":
    file = "data.train"
    lst = reading(file)
    data,labels = next(data_generator(lst,32,105))
    exit()

