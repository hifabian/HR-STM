# Note: It's not very difficult to proof the orders of these methods.
# While the second derivative is not optimal, the order itself isn't 
# changed.

import numpy as np
import scipy.interpolate as sp
import matplotlib.pyplot as plt



# Test function
f = lambda x: np.exp(np.sin(x))
df = lambda x: np.exp(np.sin(x))*np.cos(x)
ddf = lambda x: np.exp(np.sin(x))*(np.cos(x)**2-np.sin(x))


n = np.arange(10,1000,10)

step = []
err0 = []
err1 = []
err2 = []

for N in n:
  step.append(2*np.pi/(N//2))
  xt = np.arange(0, 2*np.pi, step[-1])
  yt = f(xt)
  dyt = np.gradient(yt,xt[1]-xt[0], edge_order=2)
  ddyt = np.gradient(dyt,xt[1]-xt[0], edge_order=2)
  yI = sp.interp1d(xt,yt)
  dyI = sp.interp1d(xt,dyt)
  ddyI = sp.interp1d(xt,ddyt)
  x = np.random.random(N)
  y = yI(x)
  dy = dyI(x)
  ddy = ddyI(x)
  err0.append(np.linalg.norm(y-f(x))/np.linalg.norm(f(x)))
  err1.append(np.linalg.norm(dy-df(x))/np.linalg.norm(df(x)))
  err2.append(np.linalg.norm(ddy-ddf(x))/np.linalg.norm(ddf(x)))

plt.figure()
plt.loglog(n, step, label=r"Interp. $dx$")
plt.loglog(n, err0, label=r"Approx. $f(x)$")
plt.loglog(n, err1, label=r"Approx. $\partial f(x)$")
plt.loglog(n, err2, label=r"Approx. $\partial^2 f(x)$")
plt.loglog(n, 1/n**2, '--k', label=r"$\mathcal{O}(n^{-2})$")
plt.legend()
plt.grid()
plt.show()
