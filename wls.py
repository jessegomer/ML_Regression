import numpy as np
import timeit
rows = 1000000
cols = 100
X = np.random.rand(rows,cols)
# original weight
w = 2.0
W = w*np.diag(np.ones(rows))
# added weight
d = 3.0
m = 2
# online regression
C = np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))
H = np.dot(np.dot(C,X.T),W)
em = np.zeros((1,rows))
em[0,m] = 1.0

# compare time
start = timeit.default_timer()
c = H[:,[m]]/w
a = X[[m],:]
f = np.dot(a,H)
f[0,m] = f[0,m]*(1+d/w)
gamma = 1.0/d + np.dot(a,c)[0,0]
Ct = C - np.dot(c,c.T)/gamma
Ht = H + d*np.dot(c,em) - np.dot(c,f)/gamma
end = timeit.default_timer()
print 'sparsity', (end-start)*1000

start = timeit.default_timer()
W[m,m] = W[m,m] + d
C_ = np.linalg.inv(np.dot(np.dot(X.T,W),X))
H_ = np.dot(np.dot(C_,X.T),W)
end = timeit.default_timer()
print 'dense', (end-start)*1000