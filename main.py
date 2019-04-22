import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import PolynomialFeatures


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
        return out


def run_torch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_set = np.zeros((4, 4), dtype=float)
    for i in range(0, data_set.shape[0]):
        for j in range(0, data_set.shape[1]):
            data_set[i][j] = i * 2 + j

    x = torch.tensor(data_set, device=device, requires_grad=True)
    print(x)

    y = x ** 4 + x ** 2 + x + 1
    y.backward(torch.ones(data_set.shape[0], data_set.shape[1], device=device, dtype=torch.float64))
    print(x.grad)

    y2 = 4 * x ** 3 + 2 * x + 1
    print(y2)

    x_train = np.asarray(np.random.rand(100, 1), dtype=np.float32)
    y_train = np.asarray(x_train ** 2 + 4 + np.random.rand(100, 1) * 0.2, dtype=np.float32).reshape(-1, 1)
    print(x_train.shape)
    print(y_train.shape)

    plt.scatter(x_train, y_train)
    plt.show()

    n_poly_degree = 2
    input_dim = 1 * n_poly_degree
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim)

    criterion = nn.MSELoss()  # Mean Squared Loss
    l_rate = 0.1
    optimiser = torch.optim.SGD(model.parameters(), lr=l_rate)  # Stochastic Gradient Descent
    poly = PolynomialFeatures(n_poly_degree, include_bias=False).fit(x_train)
    print('Polynomial Features:', poly.get_feature_names())

    epochs = 20000
    x_train_poly = poly.fit_transform(x_train)
    inputs = torch.from_numpy(x_train_poly)
    labels = torch.from_numpy(y_train)

    for epoch in range(epochs):
        epoch += 1
        # increase the number of epochs by 1 every time

        # clear grads as discussed in prev post
        optimiser.zero_grad()
        # forward to get predicted values
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # back props
        optimiser.step()  # update the parameters
        if epoch % 1000 == 0 or epoch == 1:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    forward_x = (np.asarray(list(range(0, 100)), dtype=np.float32) / 100).reshape(-1, 1)
    forward_x_poly = poly.fit_transform(forward_x)
    predicted = model.forward(torch.from_numpy(forward_x_poly)).data.numpy()
    plt.plot(x_train, y_train, 'go', label='from data', alpha=.5)
    plt.plot(forward_x, predicted, label='prediction', alpha=0.5)
    plt.legend()
    plt.show()
    print(model.state_dict())


def main():
    run_torch()


if __name__ == '__main__':
    main()
