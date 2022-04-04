class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.grid_x = 5
        self.grid_y = 7
    
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
        if self.x == self.grid_x-1 and self.y == self.grid_y -1:
            return True
        else:
            return False
        
    def move_left(self):
        if self.y == 3 and self.x in [0,1,2]:
            pass
        elif self.y == 5 and self.x in [2,3,4]:
            pass
        else:
            self.y = max(0, self.y-1)
    
    def move_right(self):
        if self.y == 1 and self.x in [0,1,2]:
            pass
        elif self.y == 3 and self.x in [2,3,4]:
            pass
        else:
            self.y = min(self.grid_y-1, self.y+1)
    
    def move_top(self):
        if self.x == 3 and self.y == 2:
            pass
        else:
            self.x = max(0, self.x-1)
    
    def move_bottom(self):
        if self.x == 1 and self.y == 4:
            pass
        else:
            self.x = min(self.grid_x-1, self.x+1)
    
    def get_state(self):
        return (self.x, self.y)
        
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)