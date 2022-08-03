import numpy as np
from sklearn.metrics import confusion_matrix
from model import Model
from visualisation import Visualisation
from image_preprocessing import Preprocessing

if __name__ == '__main__':
    # import objects
    visualisation = Visualisation()
    preprocessing = Preprocessing()
    model = Model()

    # Initialise parameters
    batch_size = 16
    epochs = 50
    image_size = 32
    shuffle = 50
    split_percentage = 0.25
    num_samples = 500
    verbose = 2
    num_classes = 7

    # preprocessing data
    preprocessing.prepare_data(num_samples, shuffle, split_percentage, image_size, num_classes)

    # compile model
    model.compile_model(image_size)

    # train model
    model.train_model(preprocessing.x_train, preprocessing.y_train,
                      preprocessing.x_test, preprocessing.y_test,
                      epochs, batch_size, verbose)

    # evaluate model
    model.evaluate_model(preprocessing.x_test, preprocessing.y_test)
    print('Test accuracy:', model.score[1])

    # plot the training and validation loss at each epoch
    visualisation.plot_loses(model.history.history["loss"], model.history.history["val_loss"], label="loss", num=1)

    # plot the training and validation accuracy at each epoch
    visualisation.plot_loses(model.history.history["acc"], model.history.history["val_acc"], label="loss", num=2)

    # Prediction on test data
    prediction = model.predict(preprocessing.x_test)
    # Convert predictions classes to one hot vectors
    prediction_classes = np.argmax(prediction, axis=1)
    # Convert test data to one hot vectors
    ground_true = np.argmax(preprocessing.y_test, axis=1)

    # plot confusion matrix
    cm = confusion_matrix(ground_true, prediction_classes)
    visualisation.plot_heat_map(cm)
