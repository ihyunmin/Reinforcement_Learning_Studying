import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Model

def true_fun(X):
    noise = np.random.rand(X.shape[0]) * 0.4 - 0.2
    return np.cos(1.5 * np.pi * X) + X + noise

def plot_results(model):
    x = np.linspace(0,5,100)
    input_x = torch.from_numpy(x).float().unsqueeze(1)
    plt.plot(x,true_fun(x),label='Truth')
    plt.plot(x,model(input_x).detach().numpy(), label='Prediction')
    plt.legend(loc='lower right', fontsize=15)
    plt.xlim((0,5))
    plt.ylim((-1,6))
    plt.grid()
    plt.show()
    
def main():
    data_x = np.random.rand(10000) * 5
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for step in range(10000):
        batch_x = np.random.choice(data_x, 32)
        batch_x_tensor = torch.from_numpy(batch_x).float().unsqueeze(1)
        pred = model(batch_x_tensor)
        
        batch_y = true_fun(batch_x)
        truth = torch.from_numpy(batch_y).float().unsqueeze(1)
        loss = torch.nn.functional.mse_loss(pred,truth)
        
        optimizer.zero_grad()
        loss.mean().backward() # gradient descent using back-propagation
        optimizer.step() # update the parameters
        
        if step % 1000 == 0:
            print(f'{step}th learning is finished')

    plot_results(model)
    
if __name__=="__main__":
    main()