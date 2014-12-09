
#!/usr/bin/python

# least absolute deviation

from gurobipy import *
#import timeit
#import numpy as np
#from scipy import linalg
from sklearn import datasets


cols = 100
informative = 100
ratio_new = 0.01

rows = 10000

#make the data and calculate parameters
data = datasets.make_regression(n_features=cols, n_samples=rows, n_informative=informative)


# Model
m = Model("lad")

# add beta the variable
beta = {}
for c in range(cols):
    beta[c] = m.addVar(lb=0.0, ub=GRB.INFINITY,name='beta'+str(c))

u = {}
for i in range(rows):
    u[i] = m.addVar(lb=0.0, ub=GRB.INFINITY, obj = 1, name='u'+str(i))

# The objective is to minimize the costs
m.modelSense = GRB.MINIMIZE

# Update model to integrate new variables
m.update()

# constraints
for i in range(rows):
    m.addConstr(
      quicksum(data[0][i,j] * beta[j] for j in range(cols)) - data[1][i] >= -u[i],
               'row'+str(i))
    m.addConstr(
      quicksum(data[0][i,j] * beta[j] for j in range(cols)) - data[1][i] <= u[i],
               'row'+str(i))

def printSolution():
    if m.status == GRB.status.OPTIMAL:
        print('\nMinError: %g' % m.objVal)
        print('\nbeta:')
        betax = m.getAttr('x', beta)
        print betax
#        for f in range(cols):
#            if beta[f].x > 0.0001:
#                print('%s %g' % (f, betax[f]))
    else:
        print('No solution')

# Solve
m.optimize()
printSolution()
