#下载tf.keras.applications包中已有的模型
#下载的默认位置时~/.keras/models/
import tensorflow as tf
import numpy as np
from images_process import universal_image_process
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(500,500,3),include_top=False)
#include_top是说whether to include the fully-connected layer at the top of the network。网络的top是否包含dense层
#参数固定
mobile_net.trainable = False

datafile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\data.pkl","rb")
labelsfile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\labels.pkl","rb")
import pickle
data = pickle.load(datafile)
#data归一化一下
data = np.array(data,dtype="float32")
data /= 255.0
labels = pickle.load(labelsfile)
print(len(labels[0]))

def network():
    model = tf.keras.Sequential([mobile_net,tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(5,activation='relu'),
                               tf.keras.layers.Dense(len(labels[0]),activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),loss='categorical_crossentropy',
                 metrics=["accuracy"])
    model.summary()
    #拟合
    #参数steps_per_epoch指的是每轮迭代的次数，因为上面设置了ds.batch()，所以不设置这个参数的话，默认就是总样本数除以batch的大小，这是很显然的
    model.fit(data,labels,epochs=3,batch_size=30,validation_split=0.1)
    #保存模型
    model.save("C:\\Users\\15216\Desktop\\项目\\tensorflow2\\demo_projects\\models\\h5\\selfdefinition_flowers_classification.h5")

img = data[78:81]
print(img)
model = tf.keras.models.load_model("C:\\Users\\15216\\Desktop\\models\\selfdefinition_flowers_classification.h5")
result = model.predict(img)
print(result)
print(np.argmax(result))
