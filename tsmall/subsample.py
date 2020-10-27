import numpy as np
class subsample():
  def __init__(self):
    self.res = None

  def sampleObservation(self, N, n = None, prop = None):
    if n is None  and prop is None:
      print("one of parameter n or prop is needed")
    
    elif  n is not None and prop is not None :
      print("only one of n and prop must be specificed, n is considered by defaut")
      self.res = np.random.choice(N, size=n)
    elif n is not None:
      if n > N :
        print("Cannot sample more obervations than the length of sample")
      else :
        self.res = np.random.choice(N, size=n)
    elif prop is not None:
      if prop > 1:
        print("Cannot sample more obervations than the length of sample")
      else :
        self.res = np.random.choice(N, size= int(N * prop))
    return self.res


  def sampleSequence(self, N, n = None, prop = None):
    if n is None  and prop is None:
      print("one of parameter n or prop is needed")
    elif  n is not None and prop is not None :
      print("only one of n and prop must be specified, n is considered by defaut")
      self.res = np.random.choice(N, size=n)
    elif n is not None:
      if  n > N/3:
        print("Cannot sample more obervations than the third of the length of sample")
      else :
        k = np.random.choice(np.arange(n, N-n))
        self.res = np.arange(k-n, k)
    elif prop is not None:
      if  prop > 1/3:
        print("Cannot sample more obervations than the third of the length of sample")
      else :
        n = int(N *prop)
        k = np.random.choice(np.arange(n, N-n))
        self.res = np.arange(k-n, k)
    return self.res
