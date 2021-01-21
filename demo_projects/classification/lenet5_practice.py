import tensorflow as tf
import numpy as np
from images_process import load_and_tag_images
model = tf.keras.models.Sequential([
    #第一层：6个5*5的卷积核，全0填充；最大池化，2*2的池化核，步长为2，padding='VALID'
    tf.keras.layers.Conv2D(filters=6,kernel_size=5,activation='sigmoid',input_shape=(192,192,3),padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    #第二层
    tf.keras.layers.Conv2D(filters=16,kernel_size=5,activation='sigmoid',padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    #拉直,将(28,28)像素的图像即对应的2维的数组转成一维的数组
    tf.keras.layers.Flatten(),
    #三层全连接网络
    #120个神经元
    tf.keras.layers.Dense(120,activation='sigmoid'),
    tf.keras.layers.Dense(84,activation='sigmoid'),
    tf.keras.layers.Dense(5,activation='softmax')
])

#编译模型
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
           loss='categorical_crossentropy',
           metrics=['accuracy'])

#训练集和训练标签
path = "C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2"
train_data,train_labels = load_and_tag_images(path,shape=[192,192,3])
print(train_data.shape,train_labels.shape)
model.fit(train_data,train_labels,batch_size=20,validation_split=0.1,epochs=1)
#model.save("C:\\Users\\15216\Desktop\\项目\\tensorflow2\\demo_projects\\models\\h5\\LeNet5.h5")

#C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\coltsFoot2\\image_0929.jpg
image = tf.io.read_file("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\coltsFoot2\\image_0929.jpg")
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [192,192])
# 对每个像素点的RGB做归一化处理
image /= 255.0
image = np.array(list(image))
image.shape = [1] + [192,192,3]

result = model.predict(image)
print(result)