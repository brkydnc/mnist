import numpy as np
import random
import json
from scipy.special import expit as sigmoid

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [np.random.randn(n, p) for p, n in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]

    def train(self, data, eta, epochs, batch_size, test_data=None):
        for e in range(epochs):
            random.shuffle(data)
            batches = [
                data[i:i+batch_size]
                for i in range(0, len(data), batch_size)
            ]

            for index, batch in enumerate(batches):
                self.update_over_batch(batch, eta)
                # print("Epoch: {0}/{1}, Batch: {2}/{3}".format(e+1, epochs, index+1, len(batches)))

            if test_data:
                accuracy = self.test(test_data)
                print("Epoch: {0}/{1}, Accuracy: {2}".format(e+1, epochs, accuracy))
            else:
                print("Epoch: {0}/{1}".format(e+1, epochs))

            self.save_model("model.json")
            # print("The model is saved after epoch #{0}.".format(e+1))

    def update_over_batch(self, batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for a, label in batch:
            del_nabla_w, del_nabla_b = self.backpropagation(a, label)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, del_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, del_nabla_b)]

        l = len(batch)
        self.weights = [w - eta / l * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta / l * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, a, label):
        acts = [a]
        zs = []
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # calculate z and a values of all the network
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            zs.append(z)
            acts.append(a)
        
        # calculate last layer's values
        delta = self.cost_derivative(acts[-1], label) * sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, acts[-2].T)
        nabla_b[-1] = delta

        # calculate hidden layers' values
        for l in range(2, len(self.sizes)):
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_prime(zs[-l])
            nabla_w[-l] = np.dot(delta, acts[-l-1].T)
            nabla_b[-l] = delta
        
        return nabla_w, nabla_b

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def cost_derivative(self, out, label):
        y = np.zeros((10,1))
        y[label] = 1.0
        return out - y

    def test(self, data):
        results = [(np.argmax(self.feedforward(a)), label) for a, label in data]
        s = sum(int(result == label) for result, label in results)
        return s / len(data)

    def save_model(self, file_name):
        weights = layers_to_lists(self.weights)
        biases = layers_to_lists(self.biases)
        model = {"sizes": self.sizes, "weights": weights, "biases": biases}

        with open(file_name, "w") as model_file:
            json.dump(model, model_file)


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def layers_to_lists(layers):
    l = []
    for layer in layers:
        l.append(layer.tolist())
    return l

if __name__ == "__main__":
    pass
