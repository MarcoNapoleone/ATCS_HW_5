import numpy as np
# ripristina alias rimossi da NumPy 1.24
for _name, _py in (('float', float), ('int', int), ('bool', bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _py)
