import sys
sys.path.append('..')
import numpy as np
import collections
from common.np import *
from common.layers import Embedding, SigmoidWithLoss
    
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None # 순전파 계산 결과를 저장
        
    def forward(self, h, idx):
        # idx는 ID의 배열, 미니배치 처리를 염두에 둠
        # correct_W: (N, vocab_dim)
        # h: (N, vocab_dim)
        # element wise multiplication 해주고, 각 token별 내적값을 구해줘야 하니 열연산
        correct_W = self.embed.forward(idx)
        out = np.sum(correct_W * h, axis=1) # out: (N,)
        self.cache = (h, correct_W)
        return out
    
    def backward(self, dout):
        h, correct_W = self.cache
        dout = dout.reshape(dout.shape[0], 1) # 행벡터로 변환. (N, 1)
        dcorrect_W = dout * h
        self.embed.backward(dcorrect_W)
        dh = dout * dcorrect_W
        return dh
    
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
            
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
            
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)
        
    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        
        if not GPU:  # == CPU
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
            
            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0  # target이 뽑히지 않게 하기 위함
                p /= p.sum()  # 다시 정규화 해줌
                negative_sample[i, :] = np.random.choice(self.vocab_size,
                                                         size=self.sample_size,
                                                         replace=False, p=p)
                
        else:
            # GPU(cupy)로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, 
                                               size=(batch_size, self.sample_size), 
                                               replace=True, p=self.word_p)
            
        return negative_sample
    
class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=.75, sample_size=5):
        # W는 추력 측 가중치
        
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        # 0번 계층이 정답을 다루는 계층
        self.params, self.grads = [], []
        
        for l in self.embed_dot_layers:
            self.params += l.params
            self.grads += l.grads
            
    def forward(self, h, target):
        batch_size = target.shape[0]
        incorrect_sample = self.sampler.get_negative_sample(target)
        
        # 정답 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32) # 정답이니까 1
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # 오답 순전파
        incorrect_label = np.zeros(batch_size, dtype=np.int32) # 오답이니까 0
        for i in range(self.sample_size):
            incorrect_target = incorrect_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, incorrect_target)
            loss += self.loss_layers[i + 1].forward(score, incorrect_label)
            
        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore) # 은닉층의 뉴런은 여러 번 복사되었으므로, 이 값을 모두 더해 준다.
        return dh
        