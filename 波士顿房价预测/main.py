import torch

#data 解析数据
import numpy as np
import re
data = []
ff = open('housing.data').readlines()
for item in ff:
    out = re.sub(r"\s{2,}"," ",item).strip()
    print(out)
    data.append(out.split(" "))

data = np.array(data).astype(np.float64) #np.float
print(data.shape)

Y = data[:,-1]
X = data[:,0:-1]

X_train = X[0:496,...]
Y_train = Y[0:496,...]
X_test = X[496:,...]
Y_test = Y[496:,...]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#net  定义网络  回归网络
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_output):
        super(Net, self).__init__()
        self.predict = torch.nn.Linear(n_feature,n_output)

    def forward(self,x):
        out = self.predict(x)
        return out
net = Net(13,1)

#loss
loss_func = torch.nn.MSELoss()

#optimiter
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)

#trainign
for i in range(10000):
    x_data = torch.tensor(X_train,dtype=torch.float32)
    y_data = torch.tensor(Y_train,dtype=torch.float32)
    pred = net.forward(x_data)
    loss = loss_func(pred,y_data)
    # print(pred.shape)
    # print(y_data.shape)
    loss = loss_func(pred,y_data) * 0.001
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("ite{};loss:{}".format(i,loss))
    print(pred[1:10])
    print(y_data[0:10])

