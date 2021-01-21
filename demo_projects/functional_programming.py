import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
def train():
    inputs = tf.keras.Input(shape=(32,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    predictions = layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 将这些数据分为10个类别，每种类别100个样本。[1,1,1...]->1;[2,2,2,...]->2;
    data = np.zeros((10000, 32), dtype="int32")
    k = 1000
    for i in range(10):
        data[i * k:(i + 1) * k] = i + 1

    np.random.shuffle(data)
    label = np.zeros((10000, 10), dtype="int32")

    for i in range(10000):
        index = data[i][0]
        label[i][index - 1] = 1

    val_data = np.zeros((20,32),dtype="int32")
    val_data[:10] += 8
    val_data[10:] += 9
    val_label = np.zeros((20, 10), dtype="int32")
    val_label[:10, 7] = 1
    val_label[10:, 8] = 1
    """
    ●tf.keras.callbacks.ModelCheckpoint：定期保存模型。
    ●tf.keras.callbacks.LearningRateScheduler：动态地改变学习率。
    ●tf.keras.callbacks.EarlyStopping：当模型在验证集上的性能不再提升时终止训练。
    ●tf.keras.callbacks.TensorBoard：使用TensorBoard来监测模型。
    """
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=30,monitor='val_loss'),
                 tf.keras.callbacks.TensorBoard(log_dir='logs')]
    model.fit(data, label,callbacks=callbacks,validation_data=(val_data,val_label), epochs=300, batch_size=200)
    model_structure = model.to_json()
    import json
    model_structure_dict = json.loads(model_structure)
    print(model_structure_dict)
    test_data = np.zeros((20, 32), dtype="int32")
    test_data += 4
    predictions = model.predict(test_data)
    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(np.sum(predictions[0]))
    model.save("C:\\Users\\15216\Desktop\\项目\\tensorflow2\\demo_projects\\models\\h5\\numbers_classcification300epoch.h5")
    """
    可以专门保存权重
    model.save_weights("weights.h5",save_format="h5")
    model.load_weights('weights.h5)
    保存模型结构
    json_string = model.to_json()
    """
def predict(number):
    model = tf.keras.models.load_model("C:\\Users\\15216\Desktop\\项目\\tensorflow2\\demo_projects\\models\\h5\\numbers_classcification.h5")

    test_data = np.zeros((1, 32), dtype="int32")
    test_data += number
    predictions = model.predict(test_data)
    print("输入是：",test_data)
    print("期待输出是：",9 if number > 10 else number -1 )
    print("实际输出是：",np.argmax(predictions[0]))


from h5_operator import h5_input_shape
model = tf.keras.models.load_model("C:\\Users\\15216\Desktop\\项目\\tensorflow2\\demo_projects\\models\\h5\\numbers_classcification.h5")
shape = h5_input_shape(model.to_json())
print(shape)