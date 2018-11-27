import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import BPNN


def loadfile():
    iris = pd.read_csv("iris.txt", header=None)
    iris = np.array(iris)
    iris_label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
   
    return iris, iris_label


if __name__ == "__main__":
    data, label = loadfile()

    net = BPNN.Net()
    net.setData(data, label)
    net.setLayers(layer_num=100, layer_points=8)

    #  train the net
    iter_num = 150
    loss_history = []
    for i in range(iter_num):
        loss, score = net.train(delta=1, reg=0.3, learning_rate=0.05)
        loss_history.append(loss)
        if i % 10 == 0:
            print("iter %d:\t" % i, "loss: ", loss)

    plt.plot(range(iter_num),loss_history)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()


