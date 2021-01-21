import json
import tensorflow as tf
import re
import numpy as np
def analysis_list(data_dict) -> []:
    key_list = []
    for data in data_dict:
        for key in data.keys():
            key_list.append(key)
    return key_list

#分析h5模型的输入数据的shape
#参数是模型的json描述
def h5_input_shape(model_json:str)-> list:
    model_structure = json.loads(model_json)
    # 通过反向预查来匹配模型输入
    pattern = re.compile('(?<=\'batch_input_shape\': ).*?\]')
    result = pattern.search(str(model_structure["config"])).group()
    # print(type(result), result)#[批大小，长，宽，通道]
    pattern2 = re.compile("\d+?(?=,|\])")
    shape = pattern2.findall(result)
    # print(type(shape), shape)#[长，宽，通道]
    shape = [int(element) for element in shape]
    # print(shape)
    return shape

def h5_resize_bs64(b64,shape):
    img = tf.image.decode_jpeg(b64)
    try:
        img = tf.image.resize(img,shape[:2])
        img /= 255.0 #归一化
    except Exception as e:
        print(e)
    print(type(img))#EagerTensor
    #将EagerTensor转化为numpy数组
    img = np.array(list(img))
    img.shape = (1,) + tuple(shape)#单张图片
    return img
