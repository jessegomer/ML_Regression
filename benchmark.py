#jesse gomer, cs545 machine learning group
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

def time_xhat_y(rows, cols, informative):
    data = datasets.make_regression(n_features=cols, n_samples=rows, n_informative=informative)
    xhat = linalg.pinv2(data[0])
    return time_function(lambda a,b: np.dot(a, b), xhat, data[1])

def update(old, xhat, delta):
    nonzy = np.nonzero(delta)[0]
    new = old + xhat[:,nonzy].dot(delta[nonzy])
    return new

def time_naive_update(rows, cols, informative, num_new):
    data = datasets.make_regression(n_features=cols, n_samples=rows, n_informative=informative)
    xhat = linalg.pinv2(data[0])
    old = xhat.dot(data[1])
    delta = np.concatenate((np.random.rand(num_new), np.zeros((rows - num_new))))
    return time_function(update, old, xhat, delta)


cols = 100
informative = 100
ratio_new = 0.01

rows = 100

print "number of columns:", cols
while rows <= 1000000:
    xhat_time = time_xhat_y(rows, cols, informative)
    update_time = time_naive_update(rows, cols, informative, int(rows*ratio_new))
    print "rows: {}\t\txhat*y: {}\t\tnaive update: {}".format(rows, xhat_time["best"], update_time["best"])
    rows *= 10
