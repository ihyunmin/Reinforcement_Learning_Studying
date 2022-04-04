import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qnet import Qnet
from replay_buffer import ReplayBuffer


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

# hyper parameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.lbl_episode = QLabel('Episode : ' + str(0))
        self.lbl_score = QLabel('Everage Score of past 20 episodes : ' + str(0.0))
        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_episode)
        vbox.addWidget(self.lbl_score)
        self.setLayout(vbox)

        self.setWindowTitle('Record')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()
    
    def change_label(self, episode):
        self.lbl_episode.setText('Episode : ' + str(episode))

    def change_score(self, score):
        self.lbl_score.setText('Everage Score of past 20 episodes : ' + str(round(score,2)))

# train
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a =  q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main(app, ex): 
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        ex.change_label(n_epi)
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi/200))
        # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            env.render()
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break
        
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)
        
        if n_epi%print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            ex.change_score(score/print_interval)
            print("n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score/print_interval, memory.size(), epsilon*100
            ))
            score = 0.0
    env.close()
    sys.exit(app.exec_())

if __name__=="__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    main(app,ex )
    