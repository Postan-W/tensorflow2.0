import tensorflow as tf
import pathlib
import numpy as np
from PIL import Image
import io
import logging
def load_and_process(path):
    image = tf.io.read_file(path)
    #将图片解码为三维矩阵
    image = tf.image.decode_jpeg(image,channels=3)
    #将图片格式统一
    image = tf.image.resize(image,[500,500])
    #对每个像素点的RGB做归一化处理
    image /= 255.0

    return image

path = "C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2"
#加载各个类别的图片并给他们标记
def load_and_tag_images(path,shape):
    data = []
    labels = []
    root = pathlib.Path(path)
    all_images = root.glob("*/*")
    all_images = list(all_images)
    all_images_str = [str(img) for img in all_images]
    for i in range(len(all_images_str)):
        image = tf.io.read_file(all_images_str[i])
        # 将图片解码为三维矩阵
        image = tf.image.decode_jpeg(image, channels=3)
        # 将图片格式统一
        image = tf.image.resize(image, shape[:-1])
        # 对每个像素点的RGB做归一化处理
        image /= 255.0
        data.append(list(image))
        labels.append(int(all_images[i].parent.name[-1]))

    data = np.array(data)
    labels = np.array(labels)
    labels = tf.keras.utils.to_categorical(labels)
    return data,labels

# data,labels = load_and_tag_images(path,[192,192,3])
# print("图片集的形状是:",data.shape)
# print("图片的具体编码是：",data[0])
# print("标签的形状是",labels.shape)
# print("标签的具体编码是:",labels)

def universal_image_process(image:bytes,shape:list)->np.array:
    print("tag:universal_image_process正在处理图片")
    image = Image.open(io.BytesIO(image))#读取图片文件(从内存中读取bytes流)
    try:
        len_shape = len(shape)
        image_array = np.array(image)
        # 如果输入张量是一维的，那么需要把图片拉平为一维的
        if len_shape == 1:
            #如果图片是3通道的RGB图片
            if len(image_array.shape) == 3:
                print("3通道图片处理成1维张量")
                #把图片resize到shape要求的尺寸大小
                quotient = int(shape[0] / 3)#这个quotient*3比shape[0]少的部分后面补齐
                image2 = image.resize((quotient,1))
                image2 = np.array(image2)#image2.shape->(1,quotient,3)
                product = 1
                for i in range(3):
                    product *= image2.shape[i]
                image2.shape = (product,)
                image2 = list(image2)
                #把不够的部分补上
                for i in range(shape[0]-product):
                    image2.append(127)#取一个中间值补上
                image2 = np.array(image2,dtype="float32")
                image2 /= 255.0
                image2.shape = (1,) + image2.shape
                return image2
            elif len(image_array.shape) == 2:
                print("1通道处理成一维张量")
                image3 = image.resize((shape[0],1))
                image3 = np.array(image3,dtype="float32")
                image3 /= 255.0
                return image3
        elif len_shape == 2:
            if len(image_array.shape) == 3:
                print("把3通道的处理成二维张量")
                image4 = image.resize((int(shape[0]*shape[1]/3),1))
                image4 = np.array(image4)
                image4.shape = (int(shape[0]*shape[1]/3)*3,)
                image4 = list(image4)
                for i in range(shape[0]*shape[1]-int(shape[0]*shape[1]/3)*3):
                    image4.append(127)
                image4 = np.array(image4,dtype="float32")
                image4.shape = (1,)+tuple(shape)
                image4 /= 255.0
                return image4
            elif len(image_array.shape) == 2:
                print("把单通道的处理成二维张量")
                image5 = image.resize((shape[1],shape[0]))
                image5 = np.array(image5,dtype="float32")
                print("图片形状为:",image5.shape)
                image5 /= 255.0
                image5.shape = (1,)+tuple(shape)
                return image5
        elif len_shape == 3:
            if len(image_array.shape) == 3:
                print("把3通道的转为3维张量")
                image6 = image.resize((shape[1],shape[0]))
                image6 = np.array(image6,dtype="float32")
                image6 /= 255.0
                image6.shape = (1,)+tuple(shape)
                return image6
            if len(image_array.shape) == 2:
                print("把单通道的转为3维张量")
                image7 = image.convert("RGB")
                image7 = image7.resize((shape[1],shape[0]))
                image7 = np.array(image7,dtype="float32")
                image7 /= 255.0
                image7.shape = (1,)+tuple(shape)
                return image7
    except Exception as e:
        print("发生了错误:",e)
