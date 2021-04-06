import numpy as np
import json
from mnist import MNIST
from network import Network

def load_model(file_name, sizes):
    """Try loading model from file. If can't, create a new one"""
    net = Network(sizes)

    print("Loading model... ")
    try:
        model_file = open("model.json", "r")
        model = json.load(model_file)
        model_file.close()

        sizes = model["sizes"]
        net.sizes = sizes
        net.weights = [
            np.array(w).reshape(n, p)
            for w, p, n in zip(model["weights"], sizes[:-1], sizes[1:])
        ]
        net.biases = [
            np.array(b).reshape(n, 1)
            for b, n in zip(model["biases"], sizes[1:])
        ]
        print("The model has been loaded.")
    except:
        print("Model couldn't be loaded, a new model is created.")

    return net 

def load_training(path):
    mndata = MNIST(path)
    mndata.gz = True
    images, labels = mndata.load_training()
    data = [(np.array(image).reshape(784, 1), label) for image, label in zip(images, labels)]
    return data

def load_testing(path):
    mndata = MNIST(path)
    mndata.gz = True
    images, labels = mndata.load_testing()
    data = [(np.array(image).reshape(784, 1), label) for image, label in zip(images, labels)]
    return data
