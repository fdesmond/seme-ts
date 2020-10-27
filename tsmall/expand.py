class expandseries():
  def __init__(self):
    self.indexset = None
    self.N = None
    self.mset = None
    self.fake = None


  def expandFit(self, n,  N = None, scale = None):
    # n : is the length of the small set
    # N : is the size of the big dataset the user want to obtain at the end of this procedure
    # scale : the user can provide scale insead of N. In this case N is equal to n x scale. Its default value is 5
    # value : this method returns a set of index which will be used to scale up the data
    if N is None and scale is None:
      self.N = 5 * n
    if N is not None and scale is not None :
      print("Only one of the parameters N and scale is required, by default N will be considered")
      self.N = N
    if scale is not None and N is None :
      self.N = scale * n
    if scale is None and N is not None :
      self.N = N

    self.mset = np.arange(3, int(n/2))

    return self.mset

    


  def expandinfer(self, small): 
    Nfakerows = self.N - small.shape[0]
    Nfakecolumns = small.shape[1]
    self.fake = np.zeros((Nfakerows, Nfakecolumns))
    Ncurrent = 0
    while Ncurrent < Nfakerows and len(np.intersect1d(self.mset, np.arange(Nfakerows - Ncurrent))) > 0:
      
      m = np.random.choice(np.intersect1d(self.mset, np.arange(Nfakerows - Ncurrent)))
       
      print(Ncurrent)
      b = np.random.choice(small.shape[0]-m, size= int(0.8 * (small.shape[0]-m)))
      bblock = np.array([small.iloc[a:(a+m), :].values for a in b ])
      self.fake[Ncurrent:(Ncurrent + m), :] = np.mean(bblock, axis=0)
      Ncurrent = Ncurrent + m 

    return self.fake[:Ncurrent, :]

toyexpand = expandseries()

toyexpand.expandFit(50)

toybig = toyexpand.expandinfer(toydatasubclean.iloc[subindex.tolist(), ])
