import tensorflow as tf
model =tf.keras.models.load_model("C:\\Users\\15216\\Desktop\\项目\\tensorflow2\\demo_projects\\models\\h5\\flowers_classification.h5")

# import pathlib
#
# test_images_root = pathlib.Path("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\cowslip")
# test_images_glob = list(test_images_root.glob("*"))
# test_images_paths = [str(path) for path in test_images_glob]
# path_ds = tf.data.Dataset.from_tensor_slices(test_images_paths)
# AUTOTUNE = tf.data.experimental.AUTOTUNE
#
# from local_utilities.images_process import load_and_process
# images_ds =path_ds.map(load_and_process,num_parallel_calls=AUTOTUNE)
#
# from local_utilities import change_range
# keras_ds = images_ds.map(lambda image:2*image-1)
import numpy as np
# keras_ds = np.array(list(keras_ds))
# print(keras_ds.shape)
from temp2 import img
from local_utilities.h5_operator import h5_input_shape,h5_resize_bs64
shape = h5_input_shape(model.to_json())
img = h5_resize_bs64(img,shape)
print(type(img),img.shape)
predictions = model.predict(img)
print(predictions[0])
print(np.argmax(predictions[0]))
print(np.sum(predictions[0]))