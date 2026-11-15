import os.path
import cv2 as cv
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import keras
from keras import layers

class ClassifierModel:
    def __init__(self, model_path, labels_path, model_checkpoints_path):
        self.__model = None
        self.__classes = None
        self.__model_path = model_path
        self.__labels_path = labels_path
        self.__model_checkpoints_path = model_checkpoints_path

        if os.path.exists(self.__model_path):
            self.__load_model(self.__model_path)
        else:
            print("No se encontró un modelo en la ruta especificada. Por favor, entrene uno.")

        if os.path.exists(self.__labels_path):
            self.__load_labels(self.__labels_path)
        else:
            print("No se encontraron los labels.")

    def __load_labels(self, labels_path):
        try:
            labels_df = pd.read_csv(labels_path)
            self.__classes = labels_df['description'].values
            print("Labels cargadas exitosamente - Keras")
        except Exception as e:
            print(e)

    def __load_model(self, model_path):
        try:
            self.__model = load_model(model_path)
            self.__model_path = model_path
            print("Modelo cargado con éxito - Keras")
        except Exception as e:
            print(e)

    def __build_model(self):
        inputs = keras.Input(shape=(180,180,3))
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Se define el residuo

        for size in [256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Residuo
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])  # Se hace back en el resituo
            previous_block_activation = x  # Se ubica el nuevo residuo

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.25)(x)
        outputs = layers.Dense(len(self.__classes), activation=keras.activations.softmax)(x)

        return keras.Model(inputs, outputs)

    def __compile(self):
        self.__model.__compile(
            optimizer=keras.optimizers.Adam(3e-4),
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[keras.metrics.BinaryAccuracy(name="acc")],
        )

    def __preprocess_image(self, image):
        image = image.resize((180, 180))
        img_array = keras.utils.img_to_array(image)
        img_array = keras.ops.expand_dims(img_array, 0)
        return img_array

    def __prepare_data(self, data_path):
        df_train = pd.read_csv(os.path.join(data_path, "dev.csv"))
        df_test = pd.read_csv(os.path.join(data_path, "dev.csv"))

        X_train = []
        for img in df_train['image_path']:
            img_path = cv.imread(img)
            X_train.append(cv.resize(img_path, (180, 180)))
        y_train = df_train['damaged']

        X_test = []
        for img in df_test['image_path']:
            img_path = cv.imread(img)
            X_test.append(cv.resize(img_path, (180, 180)))
        y_test = df_test['damaged']

        X_train = tf.convert_to_tensor(X_train)
        X_test = tf.convert_to_tensor(X_test)
        y_train = tf.convert_to_tensor(to_categorical(y_train))
        y_test = tf.convert_to_tensor(to_categorical(y_test))

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
        val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

        return train_ds, val_ds

    def __format_response(self, predictions):
        predictions = predictions.flatten()
        output_predictions = {label: score for label, score in zip(self.__classes, predictions)}
        return output_predictions

    def train(self, path, epochs=10):
        train_data, val_data = self.__prepare_data(path)

        if self.__model is None:
            self.__model = self.__build_model()
            self.__compile()

        callbacks = [
            keras.callbacks.ModelCheckpoint(self.__model_checkpoints_path)
        ]

        self.__model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=val_data)

    def save_model(self):
        try:
            self.__model.save(self.__model_path)
            print("Model saved at: " + self.__model_path)
        except Exception as e:
            print(e)

    def predict(self, image):
        if self.__model is None:
            return "No hay un modelo cargado, por favor, cargue uno"

        predict_image = self.__preprocess_image(image)
        predictions = self.__model.predict(predict_image)
        # predicted_indices = predictions[0].argmax(axis=-1)
        # confidence_scores = predictions[0].max(axis=-1)

        output_predictions = self.__format_response(predictions)

        return output_predictions

