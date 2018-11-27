from util import *

"""
Hidden Layer
    forward:        input -> [linear -> batch-normalize -> activate] -> output
    backward:       dout -> [actBack -> BNBack -> lineBack] -> dx
    update:         gradient descent
    
    X shape(N, in_size)
    W shape(in_size, out_size)
    b shape(1, out_size)
    out shape(N, out_size)
"""


class Layer:
    def __init__(self, in_size, out_size):
        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size)
        self.b = np.zeros(out_size)#/np.sqrt(in_size)
        self.dw = np.zeros((in_size, out_size))
        self.db = np.zeros(out_size)
        self.BN = BN()
        self.activate = Activate()


    def forward(self, x):
        #  linear
        tmp = np.dot(x, self.W) + self.b
        tmp = tmp.astype(float)

        #  BN
        norm = self.BN.forward(tmp)

        #  Activate
        # 1.sigmoid
        #  out = self.activate.sigmoidFoward(norm)

        # 2.tanh
        #  out = self.activate.tanhFoward(norm)

        # 3.ReLU
        out = self.activate.ReLUFoward(norm)

        self.cache = x

        return out

    def backward(self, dout):
        x = self.cache

        # 1.sigmoid
        #dnorm = self.activate.sigmoidBackward(dout)

        # 2.tanh
        dnorm = self.activate.tanhBackward(dout)

        # 3.ReLU
        dnorm = self.activate.ReLUBackward(dout)

        dtmp = self.BN.backward(dnorm)

        self.dw = np.dot(x.T, dtmp)
        self.db = np.sum(dtmp, axis=0)
        dx = np.dot(dtmp, self.W.T)
        return dx

    def update(self, lr):
        self.W = self.W - self.dw * lr
        self.b = self.b - self.db * lr


"""
BPNN Net
    setData:        set train data
    setLayer:       set amount of hidden layers and hidden points
    train:          train the net
"""


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

        #  score shape - (N, class)
        score = np.dot(prev, self.W)

        """
        Hinge Loss  ->   loss = 1/2 * w^2 + SUM(max(s_i - correct + delta, 0))
        """
        #  calculate hinge loss
        N, C = score.shape
        loss = 0
        dW = np.zeros_like(self.W)
        dprev = np.zeros(prev.shape)
        for i in range(N):
            correct_idx = self.target[i].astype(int)
            correct_score = score[i, correct_idx]
            for j in range(C):
                if j == correct_idx:
                    continue
                margin = score[i, j] - correct_score + delta
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

        #  update W
        self.W -= dW * learning_rate

        #  update hidden layers
        for i in range(layerNum, 0, -1):
            dprev = hiden[i-1].backward(dprev)
            hiden[i - 1].update(learning_rate)
        self.hidden_layers = hiden

        return loss, score


