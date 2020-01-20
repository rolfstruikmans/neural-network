from PIL import Image
import sys, os
import ANN
import numpy as np
import random
import json

def read_file(file_name):
    data = []
    try:
        with open(file_name, mode = 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        print('file', file_name, 'not found')
    return data

def save_model(file_name, weights, biases):
    f = open(file_name, "w")
    f.write(json.dumps({"weights": weights, "biases": biases}))
    f.close

def load_model(file_name, network):
    try:
        with open(file_name, mode = 'r') as f:
            json_string = f.read()
    except FileNotFoundError:
        print('file', file_name, 'not found')
    model = json.loads(json_string)
    network.weights = model["weights"]
    network.biases = model["biases"]

def create_mini_batch_vectors(number_of_items, batch_size):
    random_sample_vector = random.sample(range(number_of_items), number_of_items)
    mini_batch_vectors = [random_sample_vector[n:n + batch_size] for n in range(0, len(random_sample_vector), batch_size)]
    return mini_batch_vectors

def create_mini_batch(activation_data, label_data, mini_batch_vector):
    activation_batch = [activation_data[n] for n in mini_batch_vector]
    label_batch = [label_data[n] for n in mini_batch_vector]
    return (activation_batch, label_batch)

def convert_activation_data(activation_buffer, activation_size):
    activation_data = []
    for i in range(len(activation_buffer)//activation_size):
        offset = i*activation_size
        activation = activation_buffer[offset: offset+activation_size]
        a = np.frombuffer(activation, dtype=np.uint8)
        a = np.reshape(a, (a.size,1))
        a = a/255
        activation_data.append(a)
    return activation_data

def convert_label_data(y_buffer):
    label_data = []
    for i in y_buffer:
        n = y_buffer[i]
        y = np.zeros((10,1))
        y[n,0] = 1
        label_data.append(y)
    return label_data

def main():
    train_label_data = read_file('NeuralNetwork/train-labels.idx1-ubyte')
    magic_number = int.from_bytes(train_label_data[0:4], 'big')
    number_of_items = int.from_bytes(train_label_data[4:8], 'big')
    train_label_data = train_label_data[8:]
    print('magic_number:',magic_number)
    print('number_of_items:',number_of_items)
    print('--------------')

    train_activation_data = read_file('NeuralNetwork/train-images.idx3-ubyte')
    magic_number = int.from_bytes(train_activation_data[0:4], 'big')
    number_of_items = int.from_bytes(train_activation_data[4:8], 'big')
    number_of_rows = int.from_bytes(train_activation_data[8:12], 'big')
    number_of_columns = int.from_bytes(train_activation_data[12:16], 'big')
    train_activation_data = train_activation_data[16:]
    print('magic_number:', magic_number)
    print('number_of_items:', number_of_items)
    print('number_of_rows:', number_of_rows)
    print('number_of_columns:', number_of_columns)

    test_label_data = read_file('NeuralNetwork/train-labels.idx1-ubyte')
    test_label_data = test_label_data[8:]

    test_activation_data = read_file('NeuralNetwork/train-images.idx3-ubyte')
    test_activation_data = test_activation_data[16:]


    offset = 0
    image_size = 784
    image_count = 20

    # for image_index in range(image_count):
    #     image = Image.new('L', (28, 28), 'white')
    #     image.frombytes(activation_data[offset:offset+image_size])
    #     offset += image_size
    #     image_name = 'image_' + str(image_index) + '.bmp'
    #     image.save('NeuralNetwork/' + image_name)

    # runs = 3
    lrate = 0.001
    batch_size = 32
    epochs = 30

    train_activation_data = convert_activation_data(train_activation_data, image_size)
    train_label_data = convert_label_data(train_label_data)
    test_activation_data = convert_activation_data(test_activation_data, image_size)
    test_label_data = convert_label_data(test_label_data)

    #f = open("NeuralNetwork/mnist_model_784_16_16_10.tsv", "w")
    #f.write("epoch\tcost\tlrate\tbatchsize\trun\n")

    #lrates = [float(x)/200 for x in range(2,21)]
    #batch_sizes = [32,64]

    network = ANN.Network([784,16,16,10])   #for each run, initialize the network
    #load_model('NeuralNetwork/model.txt', network)

    for epoch in range(epochs):
        cost = 0
        for i in range(100):
            cost += network.cost(test_activation_data[i], test_label_data[i])
        cost /= 100

        print('epoch:', epoch, 'cost:', cost, 'lrate:', lrate, 'batch-size:', batch_size)

        mini_batch_vectors = create_mini_batch_vectors(number_of_items, batch_size)
        for mini_batch_vector in mini_batch_vectors:
            activations, labels = create_mini_batch(train_activation_data, train_label_data, mini_batch_vector)
            for a, y in zip(activations, labels):
                network.forward_pass(a)
                network.backward_pass(y)
            network.update_weights(lrate, len(mini_batch_vector))
            network.update_biases(lrate, len(mini_batch_vector))

        #f.write(str(epoch+1)+"\t"+str(cost)+"\t"+str(lrate)+"\t"+str(batch_size)+"\t"+str(run+1)+"\n")
    #f.close()
    #save_model('NeuralNetwork/model.txt', network.weights, network.biases)

    for i in range(3):
        print(network.forward_pass(test_activation_data[i]))
        print(network.layers[0].update_activation(test_activation_data[i]))
        print(test_label_data[i])
        print(network.cost(test_activation_data[i], test_label_data[i]))
        print("----------------")

if __name__ == '__main__':
     main()
