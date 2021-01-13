#下载tf.keras.applications包中已有的模型
#下载的默认位置时~/.keras/models/
import tensorflow as tf
#导入数据集
from base_process import all_image_paths,image_label_ds,AUTOTUNE,label_list
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False)
#参数固定
mobile_net.trainable = False

#使用DataSet类的shuffle方法打乱数据集
image_count = len(all_image_paths)
ds =image_label_ds.shuffle(buffer_size=image_count)

#让数据集重复几次,是指一个epoch重复几次整个数据集。不调用这个函数则默认一次
ds = ds.repeat(1)
#设置每个批次的大小,我这里有340张图片
BATCH_SIZE = 32

ds = ds.batch(BATCH_SIZE)
#通过prefetch的方法让模型的训练和每个批次的数据加载并行
"""
使用“tf.data.Dataset.prefetch”方法让ELT过程中的“数据准备和预处理（EL）”
和“数据消耗（T）”过程并行
"""
ds = ds.prefetch(buffer_size=AUTOTUNE)

"""
由于“MobileNetV2”模型接收的输入数据是归一化在[-1，1]之间的数据，
而在image_process()中对数据进行了一次归一化处理后，其范围是[0，1]，所以需要将数据映射到[-1，1]
"""
from local_utilities import change_range
keras_ds = ds.map(change_range.change_range)

"""
接下来定义模型，由于预训练好的“MobileNetV2”返回的数据维度为“（32，6，6，1280）”，其中“32”是一个批次（Batch）数据的大小，“6，6”代表输出的特征图的大小为6×6，“1280”代表该层使用了1280个卷积核。为了适应花朵分类任务，需要在“MobileNetV2”返回数据的基础上再增加两层网络层
"""
#定义线性堆叠模型结构
model = tf.keras.Sequential([mobile_net,tf.keras.layers.GlobalAveragePooling2D(),
                           tf.keras.layers.Dense(len(label_list),activation='softmax')])

"""
解释上面的网络结构：卷积层(输入层)，池化层，密集连接层(输出层）,因为是分类任务，输出层是的
神经元个数是所有花的种类数
"""
"""
全局平均池化（Global Average Pooling，GAP）是对每一个特征图求平均值，
将该平均值作为该特征图池化后的结果，因此经过该操作后数据的维度变为（32，1280）。
由于花朵分类任务是一个17分类的任务，因此需要再使用一个全连接（Dense），将维度变为（32，17）
"""

#编译模型

model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',
             metrics=["accuracy"] )

model.summary()
#拟合
#参数steps_per_epoch指的是每轮迭代的次数，因为上面设置了ds.batch()，所以不设置这个参数的话，默认就是总样本数除以batch的大小，这是很显然的
model.fit(ds,epochs=1)
#保存模型
model.save("C:\\Users\\15216\Desktop\\项目\\tensorflow2\\demo_projects\\models\\h5\\flowers_classification.h5")