import tensorflow as tf
import requests
from temp2 import img
import io
import numpy as np
import pathlib
from PIL import Image
img = Image.open(io.BytesIO(img))#读取内存中的二进制数据流

root_path = "C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2"

"""
比如在image文件夹下有5个子文件夹，每个子文件夹代表一种类型的数据，子文件夹的命名方式举例：flowers0、flowers1、
flowers2...。这样的话，下面这个函数就可以返回两个numpy数组data和labels，也可以叫张量，即每张图片的numpy表示和
其对应的onehot标记。这两个numpy数组可以直接作为网络fit时的data和labels
"""
def process_data_to_tensor(root_path:str,tensor_shape:list):
    root = pathlib.Path(root_path)
    all_images = root.glob("*/*")
    all_images = list(all_images)
    all_images = [str(img_path) for img_path in all_images]
    data = []
    labels = []
    for i in range(len(all_images)):
        with open(all_images[i], "rb") as f:
            img = f.read()
            img = Image.open(io.BytesIO(img))
            img = img.resize((tensor_shape[1],tensor_shape[0]))
            img = list(np.array(img))
            data.append(img)
            tag = int(pathlib.Path(all_images[i]).parent.name[-1])
            labels.append(tag)
    data = np.array(data)
    labels = np.array(labels)
    labels = tf.keras.utils.to_categorical(labels)
    return data,labels

data,labels = process_data_to_tensor(root_path,[500,500])
# datafile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\data.pkl","wb")
# labelsfile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\labels.pkl","wb")
# import pickle
# pickle.dump(data,datafile)
# datafile.close()
# pickle.dump(labels,labelsfile)
# labelsfile.close()
print(labels.shape)
print(data.shape)
print("第一个onehot标签是：",labels[0])
print("第81个onehot标签是：",labels[80])

# datafile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\data.pkl","rb")
# labelsfile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\labels.pkl","rb")
#
# import pickle
# loaddata = pickle.load(datafile)
# loadlabels = pickle.load(labelsfile)
# print(loaddata.shape)
# print(loadlabels.shape)