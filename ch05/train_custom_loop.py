import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm
from common.trainer import RnnlmTrainer

# hyperparams
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = .1
max_epoch = 100

# dataset
corpus, word2id, id2word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

# 바로 다음 단어를 맞춰보자
xs = corpus[:-1] # 입력
ts = corpus[1:] # 출력
data_size = len(xs)
print(f"말뭉치 크기: {corpus_size}, 어휘 수: {vocab_size}, 데이터 크기: {data_size}")

# 학습 기록
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# model, optmizer
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 각 미니배치의 시작위치 계산
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]
# 각 미니배치가 데이터를 읽기 시작하는 위치

print(f"max_iters: {max_iters}, time_size: {time_size}, batch_size: {batch_size}")

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 미니배치 획득
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size] 
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
    
    ppl = np.exp(total_loss / loss_count)
    print(f"| 에폭 {epoch + 1} | 퍼플렉시티 {ppl}")
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# training
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)
trainer.fit(xs, ts, max_epoch, batch_size, time_size)