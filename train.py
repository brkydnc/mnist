from loader import load_model, load_training, load_testing

test = True

if test:
    test_data = load_testing('./data')
data = load_training('./data')
net = load_model("model.json", [784, 100, 10])
# the model saves itself after each epoch
net.train(data, 0.21, 30, 10, test_data=None)
