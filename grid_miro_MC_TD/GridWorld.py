class GridWorld():
    def __init__(self, grid_size):
        self.x = 0
        self.y = 0
        self.grid_size = grid_size
    
    def step(self, a):
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_top()
        elif a == 2:
            self.move_right()
        elif a == 3:
            self.move_bottom()

        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done
        
    def is_done(self):
        if self.x == self.grid_size-1 and self.y == self.grid_size -1:
            return True
        else:
            return False
        
    def move_left(self):
        self.y = max(0, self.y-1)
    
    def move_right(self):
        self.y = min(self.grid_size-1, self.y+1)
    
    def move_top(self):
        self.x = max(0, self.x-1)
    
    def move_bottom(self):
        self.x = min(self.grid_size-1, self.x+1)
    
    def get_state(self):
        return (self.x, self.y)
        
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)