from loader import load_model, load_testing

data = load_testing('./data')
net = load_model("model.json", [784, 16, 10])
accuracy = net.test(data)
print("Accuracy: {0} over {1} samples".format(accuracy, len(data)))
