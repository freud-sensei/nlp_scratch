import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        self.in1 = MatMul(W_in)
        self.in2 = MatMul(W_in)
        self.out = MatMul(W_out)
        self.loss = SoftmaxWithLoss()
        
        layers = [self.in1, self.in2, self.out]
        self.params, self.grads = [], []
        for l in layers:
            self.params += l.params
            self.grads += l.grads
            
        self.word_vecs = W_in # 단어의 분산표현은 이 행렬이 된다
        
    def forward(self, contexts, target):
        h1 = self.in1.forward(contexts[:, 0])
        h2 = self.in2.forward(contexts[:, 1])
        # contexts는 3차원 행렬이다. 0~2번째 중 1번째 차원이 맥락의 윈도우 크기가 된다.
        h = (h1 + h2) * 0.5
        score = self.out.forward(h)
        loss = self.loss.forward(score, target)
        return loss
    
    def backward(self, dout=1):
        ds = self.loss.backward(dout)
        da = self.out.backward(ds)
        da *= 0.5
        self.in1.backward(da)
        self.in2.backward(da)
        return None

    