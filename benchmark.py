#cs545 machine learning group
import timeit
import numpy as np
from scipy import linalg
from sklearn import datasets


def time_function(f, *args):
    results = []
    for i in range(100):
        start = timeit.default_timer()
        f(*args)
        end = timeit.default_timer()
        results.append((end - start)*1000)
    return {'median': np.median(results), 'best':min(results)}

def make_beta(xhat, y):
    return xhat.dot(y)

def naive_update(old, xhat, delta):
    return old + xhat.dot(delta)

def update_with_nonzero(old, xhat, delta):
    nonzy = np.nonzero(delta)[0]
    new = old + xhat[:,nonzy].dot(delta[nonzy])
    return new

def make_delta(num_new, total_rows):
    head = np.random.rand(num_new)*100
    tail = np.zeros(total_rows - num_new)
    delta = np.concatenate((head, tail))
    np.random.shuffle(delta)
    return delta

def predict_y(beta, features):
    return features.dot(beta)


cols = 100
informative = 100
ratio_new = 0.01

rows = 100

print "number of columns:", cols
while rows <= 1000000:
    #make the data and calculate parameters
    data = datasets.make_regression(n_features=cols, n_samples=rows, n_informative=informative)
    num_new = int(ratio_new * rows)
    xhat = linalg.pinv2(data[0])
    delta = make_delta(num_new, rows)
    old = xhat.dot(data[1])

   #run the tests
    make_beta_time = time_function(make_beta, xhat, data[1])
    update_time = time_function(update_with_nonzero, old, xhat, delta)
    predict_y_time = time_function(predict_y, old, data[0])

    print "rows: {}\t\txhat*y: {}\t\tupdate: {}\t\tpredict: {}".format(rows, make_beta_time["best"], update_time["best"],
                                                                       predict_y_time['best'])

    rows *= 10
