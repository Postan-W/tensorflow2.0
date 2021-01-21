import tensorflow as tf
import requests
from temp2 import img
import io
import numpy as np
import pathlib
from PIL import Image
img = Image.open(io.BytesIO(img))#读取内存中的二进制数据流

root_path = "C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2"

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
# # datafile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\data.pkl","wb")
# # labelsfile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\labels.pkl","wb")
# # import pickle
# # pickle.dump(data,datafile)
# # datafile.close()
# # pickle.dump(labels,labelsfile)
# # labelsfile.close()
# print(labels.shape)
# print(data.shape)
# print("第一个onehot标签是：",labels[0])
# print("第81个onehot标签是：",labels[80])
#
# # datafile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\data.pkl","rb")
# # labelsfile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\labels.pkl","rb")
# #
# # import pickle
# # loaddata = pickle.load(datafile)
# # loadlabels = pickle.load(labelsfile)
# # print(loaddata.shape)
# # print(loadlabels.shape)





