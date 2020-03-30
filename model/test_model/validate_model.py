# ------------------------------------------------------- #
# Author: William Svea-Lochert
# Date written: 01.03.2020
# test with testset and write to log
# ------------------------------------------------------- #
from keras.models import load_model
from matrix.ConfusionMatrix import ConfusionMatrix
from termcolor import colored


# TODO: write about in git read me
def predict(model_location, categories, X, y):

    List = categories  # categories

    x_test = X
    y_test = y

    model = load_model(model_location)  # load the model
    y_new = model.predict_classes(x_test)  # Prediction

    matrix = ConfusionMatrix(List)

    for i in range(len(x_test)):  # check the predictions
        matrix.add_pair(predicted_category=List[y_new[i]], actual_category=List[y_test[i]])

    print(matrix)

    # Prints loss function first and then acc on the testset
    score = model.evaluate(x_test, y_test, verbose=0)
    text = colored('-- Loss and accuracy on testset --', 'green')
    print(text)
    print(score)
