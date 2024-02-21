import sys
sys.path.append('..')
import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        
        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f') # embedding 계층을 이용하기 때문에, 단어백터가 행 방향에 배치
        
        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=.75, sample_size=5)
        
        # 가중치, 기울기
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for l in layers:
            self.params += l.params
            self.grads += l.grads
            
        self.word_vecs = W_in # 단어의 분산표현은 여전히 이 행렬이 된다
        
    def forward(self, contexts, target):
        # contexts와 target은 원핫 벡터로 변환되지 않는다.
        # contexts는 2차원, target은 1차원
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss 
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for l in self.in_layers:
            l.backward(dout)
        return None
