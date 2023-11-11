# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:44:39 2022

@author: mofeli
"""

from PPG_SQA import PPG_SQA



from sklearn import preprocessing
import numpy as np
from scipy.signal import resample
import neurokit2 as nk


import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader



    

# def GAN_model():
#     use_gpu = torch.cuda.is_available()

#     #encoder
#     #decoder
    
#     #Generator = encoder + decoder + encoder
#     #Discriminator = encoder
    
#     batch_size = 256
    
#     encoder1 = nn.Sequential(
#         nn.Conv1d(1,32, kernel_size=4, stride = 2, padding = 0,bias=False, padding_mode='replicate'),
#         nn.ReLU(),
#         nn.Conv1d(32,64, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
#         nn.InstanceNorm1d(64),
#         nn.ReLU(),
#         nn.Conv1d(64,128, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
#         nn.InstanceNorm1d(128),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Conv1d(128,128, kernel_size = 4, stride =2,padding = 0, bias=False),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Conv1d(128,128, kernel_size = 4, stride =2, padding=1, bias=False),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Conv1d(128,128, kernel_size = 4, stride =2, padding=1, bias=False),
#         nn.ReLU(),
#         nn.Dropout(0.5),    
        
#     )
    
    
#     encoder2 = nn.Sequential(
#         nn.Conv1d(1,32, kernel_size=4, stride = 2, padding = 0,bias=False, padding_mode='replicate'),
#         nn.ReLU(),
#         nn.Conv1d(32,64, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
#         nn.InstanceNorm1d(64),
#         nn.ReLU(),
#         nn.Conv1d(64,128, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
#         nn.InstanceNorm1d(128),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Conv1d(128,128, kernel_size = 4, stride =2,bias=False),
#         nn.ReLU(),
#         nn.Dropout(0.5)
        
#     )
    
#     encoder_d = nn.Sequential(
#         nn.Conv1d(1,32, kernel_size=4, stride = 2, padding = 0,bias=False, padding_mode='replicate'),
#         nn.ReLU(),
#         nn.Conv1d(32,64, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
#         nn.InstanceNorm1d(64),
#         nn.ReLU(),
#         nn.Conv1d(64,128, kernel_size = 4, stride =2, padding =0,bias=False, padding_mode='replicate'),
#         nn.InstanceNorm1d(128),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Conv1d(128,128, kernel_size = 4, stride =2,bias=False),
        
#         nn.Sigmoid()
#     )
        
    
    
#     decoder = nn.Sequential(
#         nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, output_padding=1, bias=False),
#         nn.InstanceNorm1d(128),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.5),
#         nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, output_padding=0, bias=False),
#         nn.InstanceNorm1d(64),
#         nn.ReLU(inplace=True),
#         nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, output_padding=0, bias=False),
#         nn.InstanceNorm1d(32),
#         nn.ReLU(inplace=True),
#     )
    
#     class Generator(nn.Module):
#         def __init__(self):
#             super(Generator, self).__init__()
    
#             self.encoder1 = encoder1
#             self.decoder = decoder
#             self.encoder2 = encoder2
    
#             self.final = nn.Sequential(
#                 nn.Upsample(scale_factor=2),
#                 nn.ConstantPad1d((1, 1), 0),
#                 nn.Conv1d(32, 1, 4, padding=1,
#                           padding_mode='replicate'),
#                 nn.ReLU(inplace = True)
#             ) 
#             self.fc1 = nn.Linear(101,100)
#         def forward(self,x):
#             latent_i = self.encoder1(x)
#             #print(latent_i.shape)
#             gen_img = self.decoder(latent_i)
#             #print(gen_img.shape)
#             fin_cov = self.final(gen_img)
#             fin_cov = fin_cov.view(256,-1)
#             y = self.fc1(fin_cov)
            
#             latent_o = self.encoder2(y.reshape(256,1,-1))
#             #print(latent_o.shape)
#             return y,  latent_o
    
#     class Discriminator(nn.Module):
    
#         def __init__(self):
#             super(Discriminator, self).__init__()
#             #model = encoder()
#             #layers = list(model.children())
    
#             self.features = encoder2
#             self.classifier = encoder_d
      
#         def forward(self, x):
#             features = self.features(x)
#             features = features
#             classifier = self.classifier(x)
#             #classifier = classifier.view(-1, 1).squeeze(1)
#             return classifier, features
    
#     D = Discriminator()
#     G = Generator()
    
    
    
#     #initiate lables
#     real_label = torch.ones(size=(batch_size,128,4), dtype=torch.float32)
#     fake_label = torch.zeros(size=(batch_size,128,4), dtype=torch.float32)
    
#     #epochs
#     num_epochs = 1000
    
#     #learnin rates
#     LR_G = 1e-4
#     LR_D = 1e-4
    
    
#     #netg.apply(weights_init)
    
#     #loss functions
#     loss_adv = nn.MSELoss()
#     loss_con = nn.L1Loss()
#     loss_enc = nn.MSELoss()
#     loss_bce = nn.BCELoss()
    
#     optimizer_G = torch.optim.Adam(G.parameters(),lr=LR_G, weight_decay=1e-4)
#     optimizer_D = torch.optim.Adam(D.parameters(),lr=LR_D, weight_decay=1e-4)
    
#     # if use_gpu:
#     #     G = G.to(device)
#     #     D = D.to(device)
#     #     loss_adv = loss_adv.to(device)
#     #     loss_con = loss_con.to(device)
#     #     loss_enc = loss_enc.to(device)
#     #     loss_bce = loss_bce.to(device)
#     #     real_label = real_label.to(device)
#     #     fake_label = fake_label.to(device)
    
#     G = torch.load("989th_5s_G_5s_1-4.pth", map_location=torch.device('cpu'))
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     return G, device




def find_peaks(ppg_segment, sampling_rate=20):
    # clean PPG signal and prepare it for peak detection
    ppg_cleaned = nk.ppg_clean(ppg_segment, sampling_rate=sampling_rate)
    # peak detection
    info  = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)
    peaks = info["PPG_Peaks"]
    return peaks



def ppg_reconstruct(generator, device, ppg, gap, sample_rate):
    
    shifting_step = 5 
    # we reconstruct 5 sec ppg
    len_rec = 5*sample_rate
    
    ppg_clean = ppg
    gap_reconstructed = []
    gap_left = len(gap)
    while(len(gap_reconstructed) < len(gap)):
        ppg_clean = preprocessing.scale(ppg_clean)
        y = np.array([np.array(ppg_clean) for i in range(256)])
        ppg_test = torch.FloatTensor(y)
        rec_test,_ = generator(ppg_test.reshape(256,1,-1).to(device))
        rec_test = rec_test.cpu()
        re = rec_test.detach().numpy()

        upsampling_rate=2
        rec_resampled = resample(re[0], len(re[0]) * upsampling_rate)
        peaks_rec = find_peaks(rec_resampled)

        ppg_resampled = resample(ppg_clean, len(ppg_clean) * upsampling_rate)
        peaks_ppg = find_peaks(ppg_resampled)

        ppg_and_rec = list(ppg_resampled[:peaks_ppg[-1]]) + list(rec_resampled[peaks_rec[0]:])
        
        # len(ppg_clean) + len(recunstructed) should be 300+100= 400
        len_ppg_and_rec = len(ppg_clean) + len(re[0])
        #Downsampling
        ppg_and_rec_down = resample(ppg_and_rec, len_ppg_and_rec)
        
        
        if gap_left < (shifting_step*sample_rate):
            gap_reconstructed = gap_reconstructed + list(ppg_and_rec_down[len(ppg_clean):int((len(ppg_clean)+gap_left))])
        else:
            # select shifting_step sec of reconstructed signal
            gap_reconstructed = gap_reconstructed + list(ppg_and_rec_down[len(ppg_clean):(len(ppg_clean)+int(shifting_step*sample_rate))])
            gap_left = gap_left - (shifting_step*sample_rate)
            
        ppg_clean = ppg_and_rec_down[int(shifting_step*sample_rate):len(ppg_clean)+int(shifting_step*sample_rate)]
    
    
    return gap_reconstructed







def ppg_reconstruction(generator, device, ppg_filt, x_reliable, gaps, sample_rate):
    
    
    
    upsampling_rate=2
    ppg_original = preprocessing.scale(ppg_filt)
    reconstructed_flag = False
    for gap in gaps:
        if len(gap) <= (15*sample_rate):
            if gap[0] >= 300:
                if set(range(gap[0]-300,gap[0])).issubset(x_reliable):
                    # print(type(ppg_filt))
                    # print(type(gap))
                    # print(len(gap))
                    gap_reconstructed = ppg_reconstruct(generator, device, ppg_filt[gap[0]-300:gap[0]],gap, sample_rate)
                    # print('reconstruction applied')
                    gap_reconstructed_res = resample(gap_reconstructed, len(gap_reconstructed)*upsampling_rate)
                    ppg_original_before_gap_res = resample(ppg_original[:gap[0]], len(ppg_original[:gap[0]])*upsampling_rate)
                    ppg_original_after_gap_res = resample(ppg_original[gap[-1]:], len(ppg_original[gap[-1]:])*upsampling_rate)

                    peaks_ppg_original_before_gap = find_peaks(ppg_original_before_gap_res)
    
                    if len(gap_reconstructed_res) >=80:
                        try:
                            peaks_gap_rec = find_peaks(gap_reconstructed_res)
                            if len(ppg_original_after_gap_res) >= 80:
                                peaks_ppg_original_after_gap = find_peaks(ppg_original_after_gap_res)
    
                                ppg_original_res = list(ppg_original_before_gap_res[:peaks_ppg_original_before_gap[-1]]) + \
                                list(gap_reconstructed_res[peaks_gap_rec[0]:peaks_gap_rec[-1]]) + \
                                list(ppg_original_after_gap_res[peaks_ppg_original_after_gap[0]:])
    
                            else:
                                ppg_original_res = list(ppg_original_before_gap_res[:peaks_ppg_original_before_gap[-1]]) + \
                                list(gap_reconstructed_res[peaks_gap_rec[0]:peaks_gap_rec[-1]]) + \
                                list(ppg_original_after_gap_res)
                        
                        except:
                            continue
                    else:
                        try:
                            if len(ppg_original_after_gap_res) >= 80:
                                peaks_ppg_original_after_gap = find_peaks(ppg_original_after_gap_res)
    
                                ppg_original_res = list(ppg_original_before_gap_res[:peaks_ppg_original_before_gap[-1]]) + \
                                list(gap_reconstructed_res) + \
                                list(ppg_original_after_gap_res[peaks_ppg_original_after_gap[0]:])
    
                            else:
                                ppg_original_res = list(ppg_original_before_gap_res[:peaks_ppg_original_before_gap[-1]]) + \
                                list(gap_reconstructed_res) + \
                                list(ppg_original_after_gap_res)
                        except:
                            continue
            
                    
                    ppg_original= resample(ppg_original_res, len(ppg_original))
                    ppg_descaled = (ppg_original*np.std(ppg_filt)) + np.mean(ppg_filt)
                    reconstructed_flag = True
                    x_reliable, gaps = PPG_SQA(ppg_descaled, sample_rate, doPlot=False)
                
    if reconstructed_flag == True:
        ppg_signal = ppg_descaled
    else:
        ppg_signal = ppg_filt
        
    return ppg_signal, x_reliable, gaps