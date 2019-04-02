import math
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np


def calculate_output(x, y):
    distance = math.sqrt(x * x + y * y)
    if distance <= 1:
        return 0.8
    else:
        return 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuronalNet:
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        self.learn_rate = 0.3
        random_lower = -0.1
        random_higher = 0.1

        self.input = np.random.rand(2, 1)
        self.hidden_weight = np.random.uniform(random_lower, random_higher, (input_size + 1) * hidden_size) \
            .reshape(input_size + 1, hidden_size)
        self.hidden = np.random.uniform(random_lower, random_higher, hidden_size + 1)
        self.output_weight = np.random.uniform(random_lower, random_higher, (hidden_size + 1) * output_size) \
            .reshape(hidden_size + 1, output_size)
        self.output = np.random.uniform(random_lower, random_higher, 1).reshape(1, 1)

    def fire(self, x, y):
        self.input = np.array([x, y, 1])

        # hidden layer
        self.hidden = np.array(
            [sigmoid(x) for x in
             np.matmul(self.input, self.hidden_weight)])
        # add bias neuron
        self.hidden = np.append(self.hidden, 1)

        # output layer
        self.output = np.array(
            [sigmoid(x) for x in
             np.matmul(self.hidden, self.output_weight)])

        return self.output[0]

    def learn(self, expected_outputs):
        new_output_weights = np.copy(self.output_weight)
        new_hidden_weights = np.copy(self.hidden_weight)

        # error gradient output
        error_gradients_output = np.array(
            [deriv_sigmoid(x) for x in np.matmul(self.hidden, self.output_weight)])

        for index, gradient in np.ndenumerate(error_gradients_output):
            error = self.output[index[0]] - expected_outputs[index[0]]
            error_gradients_output[index] *= error

        # output layer
        for index, weight in np.ndenumerate(self.output_weight):
            index_hidden = index[0]
            index_output = index[1]

            delta_weight = -self.learn_rate * error_gradients_output[index_output] * self.hidden[index_hidden]
            new_output_weights[index_hidden][index_output] += delta_weight

        # error gradient hidden
        error_gradients_hidden = np.array(
            [deriv_sigmoid(x) for x in np.matmul(self.input, self.hidden_weight)])

        for index_hidden, gradient in np.ndenumerate(error_gradients_hidden):
            sum_error_gradients_output = 0
            for index_output, error_gradient in np.ndenumerate(error_gradients_output):
                sum_error_gradients_output += error_gradient * self.output_weight[index_hidden][index_output]

            error_gradients_hidden[index_hidden] *= sum_error_gradients_output

        # hidden layer
        for index, weight in np.ndenumerate(self.hidden_weight):
            index_input = index[0]
            index_hidden = index[1]

            delta_weight = -self.learn_rate * error_gradients_hidden[index_hidden] * self.input[index_input]
            new_hidden_weights[index_input][index_hidden] += delta_weight

        self.output_weight = new_output_weights
        self.hidden_weight = new_hidden_weights


DATA_SIZE = 500000
TRAIN_BEFORE_VISUALIZE = 20000
SIZE_VISUAL = 5000

all_outputs_x = []
all_outputs_y = []
all_outputs_z = []
all_values = []

n = NeuronalNet()

for i in range(0, DATA_SIZE):
    input1 = np.random.random() * 2.44 - 1.22
    input2 = np.random.random() * 2.44 - 1.22
    expected_output = calculate_output(input1, input2)
    actual_output = n.fire(input1, input2)
    n.learn([expected_output])

    if i % TRAIN_BEFORE_VISUALIZE == 0:
        z = int(i / TRAIN_BEFORE_VISUALIZE)
        output_x = []
        output_y = []
        output_z = []
        output_value = []

        for i in range(0, SIZE_VISUAL):
            inputx = np.random.random() * 2.44 - 1.22
            inputy = np.random.random() * 2.44 - 1.22
            output = n.fire(inputx, inputy)

            output_x.append(inputx)
            output_y.append(inputy)
            output_value.append(output)
            output_z.append(z)

        sc = plt.scatter(output_x, output_y, c=output_value, s=20)
        plt.colorbar(sc)
        plt.show()

        # if z >= 3:
        #     del all_outputs_x[:SIZE_VISUAL]
        #     del all_outputs_y[:SIZE_VISUAL]
        #     del all_outputs_z[:SIZE_VISUAL]
        #     del all_values[:SIZE_VISUAL]
        #
        # all_outputs_x += output_x
        # all_outputs_y += output_y
        # all_outputs_z += output_z
        # all_values += output_value
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # sc = ax.scatter(all_outputs_x, all_outputs_y, all_outputs_z, c=all_values, s=20)
        # plt.colorbar(sc)
        # plt.show()
