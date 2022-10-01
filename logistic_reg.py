import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data= datasets.load_breast_cancer()

X,y=data.data,data.target

X.shape,y.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32)).view(y_train.shape[0],1)
y_test=torch.from_numpy(y_test.astype(np.float32)).view(y_test.shape[0],1)

class LogisticRegression(nn.Module):
  def __init__(self,n_input_features):
    super(LogisticRegression,self).__init__()
    self.linear=nn.Linear(n_input_features,1)

  def forward(self,x):
    y_pred=torch.sigmoid(self.linear(x))
    return y_pred

model=LogisticRegression(X.shape[1])

criterion=nn.BCELoss()

learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

epochs=1000
for epoch in range(1,epochs+1):
  y_pred=model(X_train)

  loss=criterion(y_pred,y_train)

  loss.backward()

  optimizer.step()

  optimizer.zero_grad()

  if epoch%100==0:
    print(f'epoch = {epoch} loss = {loss.item():.5f}')

with torch.no_grad():
  y_pred=model(X_test).round()
  acc=accuracy_score(y_test,y_pred)
  print(acc)
