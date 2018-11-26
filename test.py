import pandas as pd
import numpy as np
import BPNN


def loadfile():
    iris = pd.read_csv("iris.txt", header=None)
    iris = np.array(iris)
    #  iris_data = iris[:, :4]
    #  iris_label = iris[:, 4]
    iris_label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
   
    return iris, iris_label


if __name__ == "__main__":
    data, label = loadfile()
    net = BPNN.Net()
    net.setData(data, label)
    net.setLayers(layer_num=100, layer_points=15)
    for i in range(100):
        loss, score = net.train(delta=1, reg=0.3, learning_rate=0.06)
        print("loss: ")
        print(loss)

