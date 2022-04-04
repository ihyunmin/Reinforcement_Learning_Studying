from GridWorld import GridWorld
from Agent import Agent


def MC():
    print('Monte Calro Learning')
    alpha = 0.0001
    grid_size = 4
    gamma = 1
    data = [[0]*grid_size for i in range(grid_size)]
    iterate = 50000
        
    grid_world = GridWorld(grid_size)
    agent = Agent()
    
    for i in range(iterate):
        (x,y) = grid_world.reset()
        history = []
        done = False
        while not done:
            action = agent.action()
            (x,y), reward, done = grid_world.step(action)
            history.append((x,y, reward))
        
        sum_return = 0
        for iter in history[::-1]:
            x, y, reward = iter
            # print(x,y,reward)
            data[x][y] = data[x][y] + alpha * (sum_return - data[x][y])
            # print(data[x][y])
            sum_return = reward + gamma * sum_return
        data[0][0] = data[0][0] + alpha * (sum_return - data[x][y])
    
    for row in data:
        print(row)

def TD():
    print('Temporal Difference Learning')
    alpha = 0.001
    grid_size = 4
    gamma = 1
    data = [[0]*grid_size for i in range(grid_size)]
    iterate = 50000
        
    grid_world = GridWorld(grid_size)
    agent = Agent()
    
    for i in range(iterate):
        (x,y) = grid_world.reset()
        done = False
        while not done:
            x, y = grid_world.get_state()
            action = agent.action()
            (x_prime,y_prime), reward, done = grid_world.step(action)
            
            data[x][y] = data[x][y] + alpha*(reward + gamma*data[x_prime][y_prime] - data[x][y])    
        
        grid_world.reset()    
    
    for row in data:
        print(row)

if __name__ == "__main__":
    MC()
    TD()