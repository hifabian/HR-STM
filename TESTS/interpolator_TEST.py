
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

from timeit import timeit

class Interpolator:
  def __init__(self, x, f):
    """!
      @param x Grid axes.
      @param f Function evaluated on complete grid.
    """
    self.x = x
    self.f = f
    self.dx = x[0][1]-x[0][0]
    self.dy = x[1][1]-x[1][0]
    self.dz = x[2][1]-x[2][0]

  def __call__(self, x, y, z):
    """!
      @brief Evaluates interpolated value at given points.

      @param x, y, z Positions.
    """
    indX = (x/self.dx).astype(int)
    indY = (y/self.dy).astype(int)
    indZ = (z/self.dz).astype(int)

    return ((self.x[0][indX+1]-x)*(                                 \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY,indZ]       \
                +(z-self.x[2][indZ])*self.f[indX,indY,indZ+1])      \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY+1,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX,indY+1,indZ+1]))   \
           +(x-self.x[0][indX])*(                                   \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX+1,indY,indZ+1])    \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY+1,indZ]   \
                +(z-self.x[2][indZ])*self.f[indX+1,indY+1,indZ+1])))\
      / (self.dx*self.dy*self.dz)


  def gradient(self, x, y, z, direct):
    """!
      @brief Evaluates gradient of interpolation in specified direciton.

      @param x, y, z Position.
      @param direct  Direction of gradient (x=0,y=1,z=2).
    """
    indX = (x/self.dx).astype(int)
    indY = (y/self.dy).astype(int)
    indZ = (z/self.dz).astype(int)

    # The lazy solution: copy-paste + adjust (could do something
    # fancy with indices but whether it's really worth the effort.)
    if direct == 0:
      return ((                                                     \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX+1,indY,indZ+1])    \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY+1,indZ]   \
                +(z-self.x[2][indZ])*self.f[indX+1,indY+1,indZ+1])) \
            -(                                                      \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY,indZ]       \
                +(z-self.x[2][indZ])*self.f[indX,indY,indZ+1])      \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY+1,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX,indY+1,indZ+1])))  \
      / (self.dx*self.dy*self.dz)

    if direct == 1:
      return ((                                                     \
              (self.x[0][indX+1]-x)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY+1,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX,indY+1,indZ+1])    \
             +(x-self.x[0][indX])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY+1,indZ]   \
                +(z-self.x[2][indZ])*self.f[indX+1,indY+1,indZ+1])) \
            -(                                                      \
              (self.x[0][indX+1]-x)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY,indZ]       \
                +(z-self.x[2][indZ])*self.f[indX,indY,indZ+1])      \
             +(x-self.x[0][indX])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX+1,indY,indZ+1])))  \
      / (self.dx*self.dy*self.dz)
    
    return ((                                                     \
              (self.x[1][indY+1]-y)*(                            \
                 (self.x[0][indX+1]-x)*self.f[indX,indY,indZ+1]     \
                +(x-self.x[0][indX])*self.f[indX+1,indY,indZ+1])    \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[0][indX+1]-x)*self.f[indX,indY+1,indZ+1]   \
                +(x-self.x[0][indX])*self.f[indX+1,indY+1,indZ+1])) \
            -(                                                      \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[0][indX+1]-x)*self.f[indX,indY,indZ]       \
                +(x-self.x[0][indX])*self.f[indX+1,indY,indZ])      \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[0][indX+1]-x)*self.f[indX,indY+1,indZ]     \
                +(x-self.x[0][indX])*self.f[indX+1,indY+1,indZ])))  \
      / (self.dx*self.dy*self.dz)

def Fz(x,y,z):
  return -np.cos(y)*np.sin(x)*np.exp(-x*y*z)*(x*y*z+1) / (x*y)**2
def Fy(x,y,z):
  return z*np.sin(x)*np.exp(-x*y*z)*(np.sin(y)-x*z*np.cos(y)) / (x**2*z**2+1)
def Fx(x,y,z):
  return -(z*np.cos(y)*np.exp(-x*y*z)*(y*z*np.sin(x)+np.cos(x))) / (y**2*z**2+1)
def f(x,y,z):
  return np.exp(-x*y*z)*np.sin(x)*np.cos(y)*z


def benchmark(N, xTest, yTest, zTest, wTest):
  xRef = np.linspace(0.0,1,N)
  yRef = np.linspace(0.0,1,N)
  zRef = np.linspace(0.0,1,N)
  ref = (xRef,yRef,zRef)
  wRef = f(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True))

  linInp = si.RegularGridInterpolator(ref, wRef, method="linear")
  ownInp = Interpolator(ref, wRef)
  graInp = Interpolator(ref, Fx(*np.meshgrid(xRef,yRef,zRef, indexing='ij', sparse=True)))

  wNorm = np.linalg.norm(wTest)

  print("Scipy:\t\t{:} seconds".format(timeit(lambda: linInp(np.array([xTest,yTest,zTest]).transpose()), number=100)))
  print("Linear:\t\t{:} seconds".format(timeit(lambda: ownInp(*(xTest,yTest,zTest)), number=100)))
  print("Derivative:\t{:} seconds".format(timeit(lambda: graInp.gradient(*(xTest,yTest,zTest,0)), number=100)))


  return [np.linalg.norm(linInp(np.array([xTest,yTest,zTest]).transpose())-wTest) / wNorm,
          np.linalg.norm(ownInp(*(xTest,yTest,zTest))-wTest) / wNorm,
          np.linalg.norm(graInp.gradient(*(xTest,yTest,zTest),0)-wTest) / wNorm]


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
labels = ["SciPy","Linear","Derivative"]

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
