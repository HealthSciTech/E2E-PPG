# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:41:36 2022

@author: mofeli
"""
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader



if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    #encoder
    #decoder
    
    #Generator = encoder + decoder + encoder
    #Discriminator = encoder
    
    batch_size = 256
    
    encoder1 = nn.Sequential(
        nn.Conv1d(1,32, kernel_size=4, stride = 2, padding = 0,bias=False, padding_mode='replicate'),
        nn.ReLU(),
        nn.Conv1d(32,64, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
        nn.InstanceNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64,128, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
        nn.InstanceNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv1d(128,128, kernel_size = 4, stride =2,padding = 0, bias=False),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv1d(128,128, kernel_size = 4, stride =2, padding=1, bias=False),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv1d(128,128, kernel_size = 4, stride =2, padding=1, bias=False),
        nn.ReLU(),
        nn.Dropout(0.5),    
        
    )
    
    
    encoder2 = nn.Sequential(
        nn.Conv1d(1,32, kernel_size=4, stride = 2, padding = 0,bias=False, padding_mode='replicate'),
        nn.ReLU(),
        nn.Conv1d(32,64, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
        nn.InstanceNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64,128, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
        nn.InstanceNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv1d(128,128, kernel_size = 4, stride =2,bias=False),
        nn.ReLU(),
        nn.Dropout(0.5)
        
    )
    
    encoder_d = nn.Sequential(
        nn.Conv1d(1,32, kernel_size=4, stride = 2, padding = 0,bias=False, padding_mode='replicate'),
        nn.ReLU(),
        nn.Conv1d(32,64, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
        nn.InstanceNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64,128, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
        nn.InstanceNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv1d(128,128, kernel_size = 4, stride =2,bias=False),
        
        nn.Sigmoid()
    )
        
    
    
    decoder = nn.Sequential(
        nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, output_padding=1, bias=False),
        nn.InstanceNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, output_padding=0, bias=False),
        nn.InstanceNorm1d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, output_padding=0, bias=False),
        nn.InstanceNorm1d(32),
        nn.ReLU(inplace=True),
    )
    
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
    
            self.encoder1 = encoder1
            self.decoder = decoder
            self.encoder2 = encoder2
    
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConstantPad1d((1, 1), 0),
                nn.Conv1d(32, 1, 4, padding=1,
                          padding_mode='replicate'),
                nn.ReLU(inplace = True)
            ) 
            self.fc1 = nn.Linear(101,100)
        def forward(self,x):
            latent_i = self.encoder1(x)
            #print(latent_i.shape)
            gen_img = self.decoder(latent_i)
            #print(gen_img.shape)
            fin_cov = self.final(gen_img)
            fin_cov = fin_cov.view(256,-1)
            y = self.fc1(fin_cov)
            
            latent_o = self.encoder2(y.reshape(256,1,-1))
            #print(latent_o.shape)
            return y,  latent_o
    
    class Discriminator(nn.Module):
    
        def __init__(self):
            super(Discriminator, self).__init__()
            #model = encoder()
            #layers = list(model.children())
    
            self.features = encoder2
            self.classifier = encoder_d
      
        def forward(self, x):
            features = self.features(x)
            features = features
            classifier = self.classifier(x)
            #classifier = classifier.view(-1, 1).squeeze(1)
            return classifier, features
    
    D = Discriminator()
    G = Generator()
    
    
    
    #initiate lables
    real_label = torch.ones (size=(batch_size,128,4), dtype=torch.float32)
    fake_label = torch.zeros(size=(batch_size,128,4), dtype=torch.float32)
    
    #epochs
    num_epochs = 1000
    
    #learnin rates
    LR_G = 1e-4
    LR_D = 1e-4
    
    
    #netg.apply(weights_init)
    
    #loss functions
    loss_adv = nn.MSELoss()
    loss_con = nn.L1Loss()
    loss_enc = nn.MSELoss()
    loss_bce = nn.BCELoss()
    
    optimizer_G = torch.optim.Adam(G.parameters(),lr=LR_G, weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(D.parameters(),lr=LR_D, weight_decay=1e-4)
    
    # if use_gpu:
    #     G = G.to(device)
    #     D = D.to(device)
    #     loss_adv = loss_adv.to(device)
    #     loss_con = loss_con.to(device)
    #     loss_enc = loss_enc.to(device)
    #     loss_bce = loss_bce.to(device)
    #     real_label = real_label.to(device)
    #     fake_label = fake_label.to(device)
    
    G = torch.load("989th_5s_G_5s_1-4.pth", map_location=torch.device('cpu'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    
    