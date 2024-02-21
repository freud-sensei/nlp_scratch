import sys
sys.path.append('..')
import numpy as np 
from two_layer_net import TwoLayerNet # model
from common.optimizer import SGD # optimizer
from dataset import spiral # dataset
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from tqdm.auto import tqdm

# 1. hyperparameters
learning_rate = 1.0
max_epoch = 300 # 학습 데이터를 모두 살펴본 시점
batch_size = 30
hidden_size = 10

# 2. setting up dataset, model, optimizer
x, y = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 3. training
N = len(x)
max_iters = N // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in tqdm(range(max_epoch)):
    # 3a. shuffling data
    idx = np.random.permutation(N) # 0부터 N-1까지의 무작위 순서
    x = x[idx]
    y = y[idx]
    
    # 3b. updating weights
    for it in range(max_iters):
        batch_x = x[it*batch_size:(it+1)*batch_size]
        batch_y = y[it*batch_size:(it+1)*batch_size]
        loss = model.forward(batch_x, batch_y)
        optimizer.update(model.params, model.grads)
        model.backward() # 앞서 짠 코드 덕분에 알아서 가중치가 갱신됨
        total_loss += loss
        loss_count += 1
    
    # 3c. showing progress
    if (it+1) % 10 == 0:
        avg_loss = total_loss / loss_count
        print(f"| 에폭 {epoch + 1:d} | 반복 {it + 1:d} / {max_iters:d} | 손실 {avg_loss:.2f}")
        loss_list.append(avg_loss)
        total_loss, loss_count = 0, 0