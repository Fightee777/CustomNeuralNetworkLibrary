from Classes.NeuralNetwork import NN
from Classes.NeuralNetwork import np
import urllib.request
import os


def download_mnist_csv(url, save_path):
    # Download the MNIST CSV files
    if (os.path.exists("mnist_dataset//mnist_train.csv") and
            os.path.exists("mnist_dataset//mnist_test.csv")):
        print("You already have downloaded files")
    else:
        urllib.request.urlretrieve(url, save_path)
        print("Download complete.")


mnist_train_url = "https://pjreddie.com/media/files/mnist_train.csv"
mnist_test_url = "https://pjreddie.com/media/files/mnist_test.csv"

mnist_train_save_path = "mnist_dataset//mnist_train.csv"
mnist_test_save_path = "mnist_dataset//mnist_test.csv"

download_mnist_csv(mnist_train_url, mnist_train_save_path)
print("mnist_train downloaded")
download_mnist_csv(mnist_test_url, mnist_test_save_path)
print("mnist_test downloaded")

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.01

# create instance of neural network
n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 1
i = 0
print(len(training_data_list))
for e in range(epochs):
    print("e = ", e)
    # go through all records in the training data set
    for record in training_data_list:
        i += 1
        if (i % 1000) == 0:
            print("i =", i)
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    i = 0

test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if label == correct_label:
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass

# calculate the performance score
scorecard_array = np.asarray(scorecard)
print("scorecard_array.sum() = ", scorecard_array.sum())
print("scorecard_array.size", scorecard_array.size)
print("performance = ", scorecard_array.sum() / scorecard_array.size)