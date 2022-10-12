import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim 
from torch.autograd import Variable
import time
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ANN(nn.Module):
    
    def __init__(self, input_dimension, output_dimension):
        super(ANN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.Sigmoid(),            
            nn.Linear(64,64),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(64,32),
            nn.Sigmoid(),
            nn.Linear(32,32),
            nn.Sigmoid(),
            nn.Linear(32,output_dimension),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class MultiGPU:
    
    def __init__(self, model):
        self.data_prep()
        self.model = model
        pass
        
    def ddp_setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
    def data_prep(self):
        
        df = pd.read_csv('data/Churn_Modelling.csv')
        
        X = df.iloc[:,3:13].values
        y = df.iloc[:,13].values
        
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        X = (ct.fit_transform(X))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        self.X_train=torch.FloatTensor(X_train)
        self.X_test=torch.FloatTensor(X_test)
        self.y_train=torch.LongTensor(y_train)
        self.y_test=torch.LongTensor(y_test)
        
    def start_multi_gpu_training(self, world_size):
        print("Training started on {0} GPUs".format(world_size))
        mp.spawn(self.train, args=(world_size,), nprocs=world_size)
        
    def train(self, rank, world_size):
        
        self.ddp_setup(rank, world_size)
        
        model = self.model.to(rank)
        
        ddp_model = DDP(model, device_ids=[rank])

        train = torch.utils.data.TensorDataset(self.X_train.to(rank), self.y_train.to(rank))
        test = torch.utils.data.TensorDataset(self.X_test.to(rank), self.y_test.to(rank))
       
        train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = False, sampler=DistributedSampler(train))
        
        test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

        running_loss_history = []
        epoch_list = []
        running_corrects_history = []
        val_running_loss_history = []
        val_running_corrects_history = []

        for e in range(100):

            running_loss = 0.0
            running_corrects = 0.0
            val_running_loss = 0.0
            val_running_corrects = 0.0

            for inputs, labels in train_loader:

                inputs = Variable(inputs).to(rank)
                labels = Variable(labels).to(rank)

                output = ddp_model(inputs)

                loss = criterion(output, labels.unsqueeze(1).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = torch.round(output).to(int).squeeze(1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                
            else:
                with torch.no_grad():
                    for val_inputs, val_labels in test_loader:
                        val_inputs = Variable(val_inputs).to(rank)
                        val_labels = Variable(val_labels).to(rank)
                        val_outputs = ddp_model(val_inputs)
                        val_loss = criterion(val_outputs, val_labels.unsqueeze(1).float())

                        val_preds = torch.round(val_outputs).to(int).squeeze(1)
                        val_running_loss += val_loss.item()
                        val_running_corrects += torch.sum(val_preds == val_labels.data)
                
            epoch_loss = running_loss/len(train_loader.dataset)
            epoch_acc = running_corrects.float()/ len(train_loader.dataset)
            running_loss_history.append(epoch_loss)
            running_corrects_history.append(epoch_acc)
            
            val_epoch_loss = val_running_loss/len(test_loader.dataset)
            val_epoch_acc = val_running_corrects.float()/ len(test_loader.dataset)
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc)
            
            print('Training on gpu {0}: Epoch {1}, loss : {2}, acc : {3} '.format(rank, e+1, round(epoch_loss,4), round(epoch_acc.item(),4)))
            print('Training on gpu {0}: Epoch {1}, val loss : {2}, val acc : {3} '.format(rank, e+1, round(val_epoch_loss,4), round(val_epoch_acc.item(),4)))
            
    
    
if __name__ == "__main__":
    
    model = ANN(12,1)
    
    world_size = torch.cuda.device_count()
    
    mgpu = MultiGPU(model)
    
    es = time.time()
    mgpu.start_multi_gpu_training(world_size)
    ee = time.time()
    
    print("Training over in {0} seconds".format(round(ee-es),2))


