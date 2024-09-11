import tensorflow as tf

print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib

# List all devices
print(device_lib.list_local_devices())