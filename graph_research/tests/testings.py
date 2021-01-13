import glob
from graph_research.core.utilities import *

p = sympy.symbols('p')
#Poly(332.0*p**15 - 1524.0*p**14 + 2640.0*p**13 - 2047.0*p**12 + 600.0*p**11, p, domain='RR')
pol1 = sympy.poly(332.0*p**15 - 1524.0*p**14 + 2640.0*p**13 - 2047.0*p**12 + 600.0*p**11)


#Poly(-6.0*p**6 + 24.0*p**5 - 33.0*p**4 + 16.0*p**3, p, domain='RR')
pol2 = sympy.poly(-6.0*p**6 + 24.0*p**5 - 33.0*p**4 + 16.0*p**3)


#Poly(31.0*p**9 - 153.0*p**8 + 288.0*p**7 - 246.0*p**6 + 81.0*p**5, p, domain='RR')
pol3 = sympy.poly(31.0*p**9 - 153.0*p**8 + 288.0*p**7 - 246.0*p**6 + 81.0*p**5)

print(Utilities.polynomial2binomial(pol3))