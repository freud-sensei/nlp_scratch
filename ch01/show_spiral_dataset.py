import sys
sys.path.append('..')
from dataset import spiral
import matplotlib.pyplot as plt

x, y = spiral.load_data()
print(x.shape)
print(y.shape) # 정답 클래스: 1, 오답 클래스: 0
# y는 열이 3개 -> 각 클래스마다 1개씩!