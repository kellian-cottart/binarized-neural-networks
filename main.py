from utils import *
import models
import torch

### GLOBAL VARIABLES ###
SEED = 2506
DATASETS_PATH = "./datasets"
BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)

    ### LOAD DATASETS ###
    mnist_train, mnist_test = mnist(DATASETS_PATH, BATCH_SIZE)

    ### CREATE DNN ###
    input_size = mnist_train.dataset.data.shape[1] * \
        mnist_train.dataset.data.shape[2]
    dnn = models.DNN([input_size, 128, 64, 10], init='gauss', device=DEVICE)

    ### TRAIN DNN ###
    dnn.train_network(mnist_train, n_epochs=10, learning_rate=0.01)

    ### TEST DNN ###
    accuracy = dnn.test(mnist_test)
    print('Accuracy: {:.2f}%'.format(accuracy*100))

    ### Use DNN to predict a single image ###
    # Load a single image
    index = 3
    data = mnist_test.dataset.data[index].unsqueeze(0).float()
    # Predict label
    predicted = dnn.predict(data)
    print('Predicted label: {}'.format(predicted.item()))
    true_label = mnist_test.dataset.targets[index]
    print('True label: {}'.format(true_label))
