import numpy as n
import scipy as s
import matplotlib as m
import sklearn as sk
import pandas as p
import tensorflow as t

result = t.__version__ >= '1.3.1' and m.__version__ >= '2.0.2' and sk.__version__ >= '0.18.2' and p.__version__[1:] >= '0.20.3' and s.__version__ >= '0.19.1' and n.version.version >= '1.13.1'

result = "Your versions are not up to date\n" if result else "Your results are up to date. You are good to go.\n"
print (result)

