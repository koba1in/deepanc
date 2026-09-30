import numpy as np

class FxLMS():
    def __init__(self, Len):
        self.Len = Len
        self.Wc = np.zeros(Len, dtype=np.float64)  
        self.Xd = np.zeros(Len, dtype=np.float64)  
        self.head = 0  
    
    def feedforward(self, Xf):

        self.head = (self.head - 1) % self.Len
        self.Xd[self.head] = Xf
        
        yt = np.dot(self.Wc[:self.Len - self.head], self.Xd[self.head:]) + \
             np.dot(self.Wc[self.Len - self.head:], self.Xd[:self.head])
        return yt
    
    def LossFunction(self, y, d):
        e = d - y
        return e**2, e
    
    def step(self, e, stepsize):
        self.Wc[:self.Len - self.head] += stepsize * e * self.Xd[self.head:]
        self.Wc[self.Len - self.head:] += stepsize * e * self.Xd[:self.head]

#------------------------------------------------------------------------------
# 実行関数
#------------------------------------------------------------------------------
def train_fxlms_algorithm(Model, Ref, Disturbance, Stepsize=0.00000005): 

    ref_np = np.asarray(Ref, dtype=np.float64).flatten()
    dist_np = np.asarray(Disturbance, dtype=np.float64).flatten()
    
    len_data = dist_np.shape[0]
    Erro_signal = np.zeros(len_data, dtype=np.float64) 

    for itera in range(len_data):
        xin = ref_np[itera]
        dis = dist_np[itera]
        
        y = Model.feedforward(xin)
        _, e = Model.LossFunction(y, dis)
        Model.step(e, Stepsize)
        
        Erro_signal[itera] = e
        
    return Erro_signal.tolist()
