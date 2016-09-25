import numpy as np
import netmodel
import netdata as dm

data = []


def data_creation(name):

    global data
    # Read the data
    data = dm.readdata(name)


def train(entries, text_widget):
    global data

    # If a file was downloaded
    if len(data) != 0:
        # Print text in a widget
        text_widget.delete(1.0, 'end')
        text_widget.insert('end', 'Model Training\n\n', 'big')

        # Form input and output data for a model
        X = data[:, :-1]
        y = data[:, -1].astype(int)

        # Separate the data into train and test datasets
        X_train, X_test, y_train, y_test = dm.cross_val(X, y, test_size=0.25)

        del X, y

        # Save the column with names
        names_test = X_test[:, 0]

        # Delete the columns with name
        X_train, X_test = dm.delete_names(X_train, X_test)

        # Array to matrix(HotEncoding)
        y_train = dm.encoding(y_train)

        if len(entries['Learning Rate'].get()) == 0:
            text_widget.insert('end', 'Learning rate was changed to optimal(0.01)\n\n', 'warning')
            epsilon = 0.01
        else:
            epsilon = float(entries['Learning Rate'].get())
        if len(entries['Number of Iterations'].get()) == 0:
            text_widget.insert('end', 'Iterations were changed to optimal(100)\n\n', 'warning')
            num_passes = 100
        else:
            num_passes = int(round(float(entries['Number of Iterations'].get()), 0))

        num_input = X_train.shape[1]  # input layer dimension
        num_output = y_train.shape[1]  # output layer dimension

        # Initialize a model
        model = netmodel.Neural(num_input, num_output, neurons_hlayer=30)

        # Build a model with a 3-dimensional hidden layer
        model.train_model(X_train, y_train, num_passes, epsilon, text_widget, reg_lambda=0.001, print_loss=True)

        # Predict result using the test data
        result = model.predict(X_test) + 1

        # Count difference between real and predicted. Print accuracy
        diff = np.subtract(result, y_test)
        text_widget.insert('end', "\nAccuracy: %d" % (100 - np.count_nonzero(diff)/diff.size*100) + "%\n\n", 'color')

        # Write a file (Format: type, name)
        with open("result.txt", 'w') as outfile:
            text_widget.insert('end', "Results are located in 'result.txt'", 'color')
            for i in range(len(names_test)):
                outfile.write(str(result[i]) + "->" + str(names_test[i]) + "\n")
    else:
        # Error message
        text_widget.delete(1.0, 'end')
        text_widget.insert('end', 'Firstly choose a file\nwith'
                           ' name: zoo.data.txt\n', 'big')
