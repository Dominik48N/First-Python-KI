import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

if __name__ == "__main__":
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = [1, 2, 3]
    ys = [2, 4, 6]

    model.fit(xs, ys, epochs=3000)

    print(model.predict([7]))
