from GridWorld import GridWorld
from Agent import QAgent
import numpy as np


def MCcontrol():
    print('Monte Calro control')

    iterate = 1000
    grid_world = GridWorld()
    agent = QAgent()
    
    for i in range(iterate):
        
        history = []
        done = False
        s = grid_world.reset()
        while not done:
            action = agent.select_action(s)
            s_prime, reward, done = grid_world.step(action)
            history.append((s,action, reward, s_prime))
            s = s_prime
        agent.update_table_MC(history)
        agent.anneal_eps()
        
        if i % 100 == 0:
            print(f'{i}th learning is finished!')
    
    agent.show_table()

def SARSA():
    print('SARSA Learning')

    iterate = 1000
    grid_world = GridWorld()
    agent = QAgent()
    
    for i in range(iterate):
        
        done = False
        s = grid_world.reset()
        while not done:
            action = agent.select_action(s)
            s_prime, reward, done = grid_world.step(action)
            agent.update_table_TD((s,action,reward,s_prime))
            s = s_prime
        agent.anneal_eps()
        
        if i % 100 == 0:
            print(f'{i}th learning is finished!')
    
    agent.show_table()

def Q():
    print('Q Learning')
    
    iterate = 1000
    grid_world = GridWorld()
    agent = QAgent()
    
    for i in range(iterate):
        
        done = False
        s = grid_world.reset()
        while not done:
            action = agent.select_action(s)
            s_prime, reward, done = grid_world.step(action)
            agent.update_table_TD((s,action,reward,s_prime))
            s = s_prime
        agent.anneal_eps()
        
        if i % 100 == 0:
            print(f'{i}th learning is finished!')
    
    agent.show_table()
    
if __name__ == "__main__":
    Q()
