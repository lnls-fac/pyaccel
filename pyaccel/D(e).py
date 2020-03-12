import numpy as np
from scipy import fixed_quad
import math


def Calc_D(eps):
    f1 = math.exp(-x)*math.log(x)/x
    f2 = math.exp(-x)/x
    I1 = fixed_quad(f1, eps, 100*eps)
    I2 = fixed_quad(f2, eps, 100*eps)
    D = math.sqrt(eps)(-3*math.exp(-eps)/2+eps*I1/2+(3*eps-eps*math.log(eps)+2)*I2
    return D

x = np.linspace(-5, 1, 3)
for i in range (len(x)):
    eps = math.exp(

