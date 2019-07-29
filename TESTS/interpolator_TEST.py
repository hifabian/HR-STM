# Note: The results here are not difficult to proof, but still nice
# to visualize.

import sys
import os
# Include directory
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

from timeit import timeit

from hrstm_tools.interpolator import Interpolator


def Fz(x,y,z):
  return y*np.sin(x)*np.exp(-x*y*z)*(np.sin(z)-x*y*np.cos(z)) / (x**2*y**2+1)
def Fy(x,y,z):
  return -np.cos(z)*np.sin(x)*np.exp(-x*y*z)*(x*y*z+1) / (x*z)**2
def Fx(x,y,z):
  return -(y*np.cos(z)*np.exp(-x*y*z)*(y*z*np.sin(x)+np.cos(x))) / (y**2*z**2+1)
def f(x,y,z):
  return np.exp(-x*y*z)*np.sin(x)*np.cos(z)*y


def benchmark(N, xTest, yTest, zTest, wTest):
  xRef = np.linspace(0.0,1,N)
  yRef = np.linspace(0.0,1,N)
  zRef = np.linspace(0.0,1,N)
  ref = (xRef,yRef,zRef)
  wRef = f(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True))

  linInp = si.RegularGridInterpolator(ref, wRef, method="linear")
  ownInp = Interpolator(ref, wRef)
  graInp = Interpolator(ref, Fx(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))
  gr2Inp = Interpolator(ref, Fz(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))

  wNorm = np.linalg.norm(wTest)

  print("Scipy:\t\t{:} seconds".format(timeit(lambda: linInp(np.array([xTest,yTest,zTest]).transpose()), number=100)))
  print("Linear:\t\t{:} seconds".format(timeit(lambda: ownInp(*(xTest,yTest,zTest)), number=100)))
  print("Derivative:\t{:} seconds".format(timeit(lambda: graInp.gradient(*(xTest,yTest,zTest,1)), number=100)))
  print("Derivative:\t{:} seconds".format(timeit(lambda: gr2Inp.gradient(*(xTest,yTest,zTest,3)), number=100)))

  """
  xRef = np.linspace(0.123,0.9,int(N*np.pi))
  yRef = np.linspace(0.123,0.9,int(N*np.pi))
  zRef = np.linspace(0.123,0.9,int(N*np.pi))
  ref = (xRef,yRef,zRef)
  fig, axes = plt.subplots(3,3)

  axes[0,0].imshow((ownInp(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))[:,:,0])
  axes[0,1].imshow((f(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))[:,:,0])
  axes[0,2].imshow(abs((ownInp(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))[:,:,0] \
   -(f(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))[:,:,0]), cmap="gist_gray")

  axes[1,0].imshow((graInp.gradient(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True),1))[:,:,0])
  axes[1,1].imshow((f(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))[:,:,0])
  axes[1,2].imshow(abs((graInp.gradient(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True),1))[:,:,0]\
   -(f(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))[:,:,0]), cmap="gist_gray")

  axes[2,0].imshow((gr2Inp.gradient(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True),3))[:,:,0])
  axes[2,1].imshow((f(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))[:,:,0])
  axes[2,2].imshow(abs((gr2Inp.gradient(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True),3))[:,:,0] \
   -(f(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))[:,:,0]), cmap="gist_gray")

  plt.show()
  """

  return [ 
          np.linalg.norm(linInp(np.array([xTest,yTest,zTest]).transpose())-wTest) / wNorm,
          np.linalg.norm(ownInp(*(xTest,yTest,zTest))-wTest) / wNorm,
          np.linalg.norm(graInp.gradient(*(xTest,yTest,zTest),1)-wTest) / wNorm,
          np.linalg.norm(gr2Inp.gradient(*(xTest,yTest,zTest),3)-wTest) / wNorm]


xTest = np.sort(np.random.random(100))
yTest = np.sort(np.random.random(100))
zTest = np.sort(np.random.random(100))
wTest = f(xTest,yTest,zTest)


n = np.arange(16,256,16)
res = []
for N in n:
  print("Number of points =", N)
  res.append(benchmark(N,xTest,yTest,zTest,wTest))

res = np.array(res)
labels = [
  "$\mathcal{I}(f)$", \
  "$\mathcal{I}(f)$", \
  "$\partial_x\mathcal{I}(f)$", #"FD$_x$+SciPy",
  "$\partial_z\mathcal{I}(f)$"]#, "FD$_z$+SciPy"]

plt.figure()
for i in range(np.shape(res)[1]):
  plt.loglog(n, res[:,i], label=labels[i])
  print(np.log(res[1:,i]/res[:-1,i]) / np.log(n[1:]/n[:-1]))
  print(np.log(res[-1,i]/res[0,i]) / np.log(n[-1]/n[0]))
plt.loglog(n, 1/n**1, '--k', label=r"$\mathcal{O}(n^{-1})$")
plt.loglog(n, 1/n**2, '--k', label=r"$\mathcal{O}(n^{-2})$")
plt.legend()
plt.grid()
plt.show()
