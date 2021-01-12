import glob
from graph_research.core.utilities import *

p = sympy.symbols('p')
#Poly(332.0*p**15 - 1524.0*p**14 + 2640.0*p**13 - 2047.0*p**12 + 600.0*p**11, p, domain='RR')
pol1 = sympy.poly(332.0*p**15 - 1524.0*p**14 + 2640.0*p**13 - 2047.0*p**12 + 600.0*p**11)


#Poly(-6.0*p**6 + 24.0*p**5 - 33.0*p**4 + 16.0*p**3, p, domain='RR')
pol2 = sympy.poly(-6.0*p**6 + 24.0*p**5 - 33.0*p**4 + 16.0*p**3)

print(Utilities.polynomial2binomial(pol2))