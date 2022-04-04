import random
import numpy as np

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5,7,4)) # 5x7 miro에서 4방향의 q(s,a) table
        self.eps = 0.9 # for e-greedy
        self.alpha = 0.01
    
    def select_action(self, s):
        # eps-greedy
        x, y = s
        coin = random.random()
        if coin < 1 - self.eps:
            action_val = self.q_table[x,y,:]
            next_action = np.argmax(action_val)
        else:
            next_action = random.randint(0,3)
            
        return next_action
    
    def update_table_MC(self, history):
        # 한 에피소드에 해당하는 history를 받아서 q table의 값을 업데이트
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x, y = s
            self.q_table[x,y,a] = self.q_table[x,y,a] + self.alpha * (cum_reward - self.q_table[x,y,a])
            cum_reward = cum_reward + r
    
    def update_table_TD(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        x_prime, y_prime = s_prime
        a_prime = self.select_action(s_prime) # 실제로 움직일 액션이 아닌 s_prime에서 선택할 액션
        # variance 가 낮아 기존 alpha보다 * 10 증가.
        self.q_table[x,y,a] = self.q_table[x,y,a] + self.alpha * 10 * ( r + self.q_table[x_prime,y_prime,a_prime] - self.q_table[x,y,a])
    
    def update_table_Q(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        x_prime, y_prime = s_prime
        # a_prime = self.select_action(s_prime) # 실제로 움직일 액션이 아닌 s_prime에서 선택할 액션
        # variance 가 낮아 기존 alpha보다 * 10 증가.
        self.q_table[x,y,a] = self.q_table[x,y,a] + self.alpha * 10 * ( r + np.amax(self.q_table[x_prime,y_prime,:]) - self.q_table[x,y,a])
            
    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)
        
    def anneal_eps_Q(self):
        self.eps -= 0.01
        self.eps = max(self.eps, 0.2)
        
    def show_table(self):
        q_list = self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_list)):
            row = q_list[row_idx]
            for col_idx in range(len(row)):
                value_list = row[col_idx]
                action = np.argmax(value_list)
                data[row_idx,col_idx] = action
        print(data)
        