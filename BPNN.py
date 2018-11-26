import numpy as np

def sigmoid(x):
    s = 1.0 / (1 + np.exp(-x))
    return s

class Layer:
    def __init__(self, in_size, out_size):
        self.W = np.random.randn(in_size, out_size)
        self.b = np.random.randn(out_size)
        self.dw = np.zeros((in_size, out_size))
        self.db = np.zeros(out_size)

    # X shape(N, A)
    # W shape(A, B)
    # b shape(1, B)
    # out shape(N, B)
    def forward(self, x):
        tmp = np.dot(x, self.W) + self.b

        # Logistic
        tmp = tmp.astype(float)
        out = sigmoid(tmp)

        self.buffer_out = out
        self.buffer_x = x

        return out

    def backward(self, dout):
        out = self.buffer_out
        x = self.buffer_x

        dtmp = out*(1-out)*dout
        self.dw = np.dot(x.T, dtmp)
        self.db = np.sum(dtmp, axis=0)
        dx = np.dot(dtmp, self.W.T)
        return dx

    def update(self, lr):
        self.W = self.W - self.dw * lr
        self.b -= self.db * lr


class Net:
    def __init__(self):
        pass

    def setData(self, input, labels):
        N, D = input.shape
        self.data = input[:, :D-1]
        target = np.zeros((N, 1))
        for i in range(N):
            label = input[i, D-1]
            target[i] = labels.index(label)
        self.labels = labels
        self.target = target

    def setLayers(self, layer_num, layer_points=2):
        N,D = self.data.shape

        hidden_layers = []
        hidden_layers.append(Layer(D, layer_points))
        for i in range(layer_num-1):
            hidden_layers.append(Layer(layer_points, layer_points))
        self.hidden_layers = hidden_layers

        self.W = np.random.randn(layer_points, len(self.labels))

    def train(self, delta=1, reg=0.3, learning_rate=0.03):
        hiden = self.hidden_layers
        layerNum = len(hiden)

        prev = hiden[0].forward(self.data)
        for i in range(1, layerNum):
            prev = hiden[i].forward(prev)

        #  prev(N, layerPoint)
        out = np.dot(prev, self.W)  #(N, laybelnum)

        #hinge loss
        N, C = out.shape
        loss = 0
        dW = np.zeros_like(self.W) #(layerPoint, labelNum)
        dprev = np.zeros(prev.shape)
        for i in range(N):
            correct_idx = self.target[i].astype(int)
            correct_score = out[i, correct_idx]
            for j in range(C):
                if j == correct_idx:
                    continue
                margin = out[i, j] - correct_score + delta
                if margin > 0:
                    loss += margin
                    dW[:, correct_idx] += -prev[i].reshape(-1, 1)
                    dW[:, j] += prev[i].T
                    h = self.W[:, j].reshape(1, -1) - self.W[:, correct_idx].reshape(1, -1)
                    for m in range(dprev.shape[1]):
                        dprev[i, m] = h[0, m]

        loss /= N
        dW /= N
        loss += 0.5 * reg * np.sum(self.W * self.W)
        dW += reg * self.W

        self.W -= dW * learning_rate
        for i in range(layerNum, 0, -1):
            dprev = hiden[i-1].backward(dprev)
            hiden[i - 1].update(learning_rate)

        self.hidden_layers = hiden
        return loss, out


