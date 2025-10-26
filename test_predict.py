# test_predict.py
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('mnist_cnn_model.h5')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x = x_test[0].astype('float32') / 255.0
x = x.reshape(1,28,28,1)
pred = model.predict(x)
digit = np.argmax(pred)
print("True:", y_test[0], "Predicted:", digit, "Confidence:", float(pred[0][digit]))
