import tensorflow as tf
from my_model import create_model


class TestModel:
    def setup_method(self):
        self.mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (
            self.x_test,
            self.y_test,
        ) = self.mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        self.model = create_model()

    def test_model_creation(self):
        assert isinstance(self.model, tf.keras.models.Sequential)
        assert len(self.model.layers) == 4
        assert isinstance(self.model.layers[0], tf.keras.layers.Flatten)
        assert isinstance(self.model.layers[1], tf.keras.layers.Dense)
        assert isinstance(self.model.layers[2], tf.keras.layers.Dropout)
        assert isinstance(self.model.layers[3], tf.keras.layers.Dense)

    def test_model_training(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=1)
        # loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        # print("loss: %s" % loss)
        # print("accuracy: %s" % accuracy)

        self.model.save("my_model.h5")
        assert history.history["accuracy"][-1] > 0.8

        # Save the model

    def test_model_evaluation(self):
        # Load the model from the training
        self.model = tf.keras.models.load_model("my_model.h5")
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print("loss: %s" % loss)
        print("accuracy: %s" % accuracy)
        assert loss < 0.5, "loss should be less than 0.5, but got %s" % loss
        assert accuracy > 0.8, (
            "accuracy should be greater than 0.8, but got %s" % accuracy
        )
