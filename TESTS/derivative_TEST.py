import sys
import os
# Include directory
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
import numpy as np
import matplotlib.pyplot as plt
from hrstm_tools.interpolator import Interpolator


# Test functions
def f(x,y,z):
  return np.exp(np.sin(x)*np.cos(y))*np.sin(z)
def dfdx(x,y,z):
  return np.cos(x)*np.cos(y)*np.sin(z)*np.exp(np.sin(x)*np.cos(y))
def dfdy(x,y,z):
  return -np.sin(x)*np.sin(y)*np.sin(z)*np.exp(np.sin(x)*np.cos(y))
def dfdz(x,y,z):
  return np.exp(np.sin(x)*np.cos(y))*np.cos(z)

# Test grid
xTest = np.sort(np.random.random(100))
yTest = np.sort(np.random.random(100))
zTest = np.sort(np.random.random(100))
rTest = (xTest,yTest,zTest)

# Exact values
wTest = f(*rTest)
xTest = dfdx(*rTest)
yTest = dfdy(*rTest)
zTest = dfdz(*rTest)
wNorm = np.linalg.norm(wTest)
xNorm = np.linalg.norm(xTest)
yNorm = np.linalg.norm(yTest)
zNorm = np.linalg.norm(zTest)

# Loop over different grid sizes
n = np.arange(10,100,10)
errors = np.empty((0,4))
for N in n:
  # Reference grid with N^3 nodes
  xRef = np.linspace(0.0,1,N)
  yRef = np.linspace(0.0,1,N)
  zRef = np.linspace(0.0,1,N)
  rRef = (xRef,yRef,zRef)
  mRef = np.meshgrid(xRef,yRef,zRef,indexing='ij',sparse=True)
  fRef = f(*mRef)
  # Interpolator
  inter = Interpolator(rRef, fRef)
  # Error output
  errors = np.vstack((errors, np.array([
    np.linalg.norm(inter(*rTest)-wTest) / wNorm,
    np.linalg.norm(inter.gradient(*rTest,1)-xTest) / xNorm,
    np.linalg.norm(inter.gradient(*rTest,2)-yTest) / yNorm,
    np.linalg.norm(inter.gradient(*rTest,3)-zTest) / zNorm,
  ])))


plt.figure()
plt.loglog(n, errors[:,0], label=r"$f(x,y,z)$")
plt.loglog(n, errors[:,1], label=r"$\partial_xf(x,y,z)$")
plt.loglog(n, errors[:,2], label=r"$\partial_yf(x,y,z)$")
plt.loglog(n, errors[:,3], label=r"$\partial_zf(x,y,z)$")
plt.loglog(n, 1/n, ":k", label=r"$\mathcal{O}(n^{-1})$")
plt.loglog(n, 1/n**2, "--k", label=r"$\mathcal{O}(n^{-2})$")
plt.legend()
plt.grid()
plt.show()
