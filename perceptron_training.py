from numpy import array, random, dot
from random import choice
from pylab import ylim, plot
from matplotlib import pyplot as plt

step_function = lambda x: 0 if x < 0 else 1

training_dataset = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

weights = random.rand(3) #initialize wieghts

error = []
learning_rate = 0.2
n = 100

for j in range(n):
    x, expected = choice(training_dataset)
    result = dot(weights, x)
    err = expected - step_function(result)
    error.append(err)
    weights += learning_rate * err * x

for x, _ in training_dataset:
    result = dot(x, weights)
    print("{}: {} -> {}".format(x[:2], result, step_function(result)))

ylim([-1,1])
plot(error)
plt.title('Error Curve')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.show()
