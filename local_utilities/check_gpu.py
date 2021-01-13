from tensorflow.python.client import device_lib

def available_devices():
    return device_lib.list_local_devices()

