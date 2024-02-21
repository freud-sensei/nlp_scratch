import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul

c1 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c2 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 가중치 초기화
W_in = np.random.randn(7, 3)
# 은닉층의 차원 수가 입력층보다 작아야 embedding의 의미가 있음!!
W_out = np.random.randn(3, 7) 

# layer 생성
in_layer1 = MatMul(W_in)
in_layer2 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파
h1 = in_layer1.forward(c1)
h2 = in_layer2.forward(c2)
h = (h1 + h2) * 0.5
s = out_layer.forward(h)

print(s)