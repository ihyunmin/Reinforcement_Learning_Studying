import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from actorcritic import ActorCritic

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

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

learning_rate = 0.0002
gamma =0.98
n_rollout = 10

def main(app, ex): 
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        ex.change_label(n_epi)
        # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            for t in range(n_rollout):
                env.render()
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))
                s = s_prime
                score += r
        
        model.train_net()
        
        if n_epi%print_interval == 0 and n_epi != 0:
            ex.change_score(score/print_interval)
            print("# of episode : {}, avg score : {:.1f}".format(
                n_epi, score/print_interval
            ))
            score = 0.0

    env.close()
    sys.exit(app.exec_())

if __name__=="__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    main(app,ex)
    