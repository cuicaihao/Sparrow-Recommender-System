# %% Build a Model
import tensorflow as tf

# load the MNIST dateset

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name="input"),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax', name="output")
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test,  y_test, verbose=2)
 

# %%  Save the model in HDF5 format for serving
model.save('model.h5')
 
# delete the model in RAM
del model

# Reload the model and test it
m = tf.keras.models.load_model('model.h5')
print(m.summary())
m.evaluate(x_test,  y_test, verbose=2)
 


# %% Generate prediction on x_test
from sklearn.metrics import accuracy_score
predictions = m.predict(x_test)
# calculate the accuracy with scikit-learn
print("Accuracy: %.2f%%" % (accuracy_score(y_test, predictions.argmax(axis=1)) * 100.0))
 
# %% run the model in onnx format
import tf2onnx
import onnxruntime as rt

spec = (tf.TensorSpec((None, 28, 28), tf.float32, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]


# %% Run the model in onnx format
providers = ['CPUExecutionProvider']
m_onnx = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m_onnx.run(output_names, {"input":tf.convert_to_tensor(x_test, dtype=tf.float32).numpy()})
print("Accuracy: %.2f%%" % (accuracy_score(y_test, onnx_pred[0].argmax(axis=1)) * 100.0))


# %%
