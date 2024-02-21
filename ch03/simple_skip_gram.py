import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        self.in = MatMul(W_in)
        self.out = MatMul(W_out)
        self.loss1 = SoftmaxwithLoss()
        self.loss2 = SoftmaxWithLoss()
        
        layers = [self.in, self.out]
        self.params, self.grads = [], []
        for l in layers:
            self.params += l.params
            self.grads += l.grads
            
        self.word_vecs1 = W_in
        self.word_vecs2 = W_out.T
        # 단어의 분산표현은 이 두 행렬이 된다
        
    def forward(self, contexts, target):
        h = self.forward(target)
        score = self.out.forward(h)
        loss1 = self.loss1.forward(score, contexts[:, 0])
        loss2 = self.loss2.forward(score, contexts[:, 1])
        loss = loss1 + loss2
        return loss
    
    def backward(self, dout=1):
        dl1 = self.loss1.backward(dout)
        dl2 = self.loss2.backward(ds)
        ds = dl1 + dl2
        dh = self.out.backward(ds)
        self.in.backward(dh)
        return None

    