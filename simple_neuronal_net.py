import math
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np


def calculate_output(x, y):
    return 0 if x * x + y * y > 1 else 0.8


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuronalNet:
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        self.hidden_size = hidden_size

        self.learn_rate = 0.1
        random_lower = -1
        random_higher = 1

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

    def fireHidden(self, x, y, whichHidden):
        self.input = np.array([x, y, 1])

        # hidden layer
        self.hidden = np.array(
            [sigmoid(x) for x in
             np.matmul(self.input, self.hidden_weight)])
        # add bias neuron
        self.hidden = np.append(self.hidden, 1)

        one_hot = np.zeros((self.hidden_size + 1, self.hidden_size + 1))
        one_hot[whichHidden][whichHidden] = 1
        output_weight_copy = np.matmul(one_hot, self.output_weight)

        # output layer
        self.output = np.array(
            [sigmoid(x) for x in
             np.matmul(self.hidden, output_weight_copy)])

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


DATA_SIZE = 1000000
TRAIN_BEFORE_VISUALIZE = 10000
SIZE_VISUAL = 70

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
        output_x = np.empty(SIZE_VISUAL * SIZE_VISUAL)
        output_y = np.empty(SIZE_VISUAL * SIZE_VISUAL)
        output_value = np.empty((7, SIZE_VISUAL * SIZE_VISUAL))

        for counter_x in range(0, SIZE_VISUAL):
            for counter_y in range(0, SIZE_VISUAL):
                inputx = 1.22 - counter_x / SIZE_VISUAL * 2.44
                inputy = 1.22 - counter_y / SIZE_VISUAL * 2.44
                output = n.fire(inputx, inputy)

                output_x[counter_x * SIZE_VISUAL + counter_y] = inputx
                output_y[counter_x * SIZE_VISUAL + counter_y] = inputy
                output_value[0][counter_x * SIZE_VISUAL + counter_y] = output

                for i in range(5):
                    outputHidden = n.fireHidden(inputx, inputy, i)
                    output_value[1 + i][counter_x * SIZE_VISUAL + counter_y] = outputHidden

        fig, ax = plt.subplots(nrows=3, ncols=3)
        sc = None

        titles = ["Output", "Hidden 0", "", "Hidden 1", "Hidden 2", "", "Hidden 3", "Bias", ""]

        for index_row, row in enumerate(ax):
            for index_col, col in enumerate(row):
                if index_col != 2:
                    sc = col.scatter(output_x, output_y, c=output_value[index_row * 2 + index_col], s=20)

                col.set_title(titles[index_row * 3 + index_col])

        fig.colorbar(sc)
        plt.show()
