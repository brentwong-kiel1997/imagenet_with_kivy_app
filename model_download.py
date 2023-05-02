import tensorflow as tf
import tensorflow_hub as hub

m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_152/classification/5")
])
m.build([None, 224, 224, 3])  # Batch input shape.

m.save('imagenet')