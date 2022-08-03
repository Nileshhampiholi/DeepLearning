from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


class Model:
    def __init__(self):
        self.model = None
        self.history = None
        self.score = None

    def compile_model(self, image_size=32):
        """
        CNN model
        """

        # Layer 1
        self.model = Sequential()
        self.model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        # Layer 2
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        # Layer 3
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        # Add dense layers
        self.model.add(Flatten())
        self.model.add(Dense(32))
        self.model.add(Dense(7, activation='softmax'))

        # Analyse model
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

    def train_model(self, x_train, y_train, x_test, y_test, epochs, batch_size, verbose):
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=verbose)

    def evaluate_model(self, x_test, y_test):
        self.score = self.model.evaluate(x_test, y_test)

    def predict(self, data):
        return self.model.predict(data)

