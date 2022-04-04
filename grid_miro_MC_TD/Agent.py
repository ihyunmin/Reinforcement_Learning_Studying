import random

class Agent():
    def __init__(self):
        pass
    
    def action(self):
        a = int(random.random()/0.25) % 4
        return a
        