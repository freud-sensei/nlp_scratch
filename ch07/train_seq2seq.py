import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq

# dataset
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char2id, id2char = sequence.get_vocab()

# hyperparams
vocab_size = len(char2id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# model and optimizer
model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 훈련 과정
acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    correct_num = 0
    
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id2char, verbose)
    
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print(f"검증 정확도: {acc * 100}")