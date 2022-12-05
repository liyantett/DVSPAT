# Add an encoder/decoder


import sys
from utils_scheduler.TrackLoss import TrackLoss
#reload(sys)
#sys.setdefaultencoding('utf-8')
import os
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import scipy
import scipy.io

from guided_back_propagation import GuidedBackPropagation


from CASME2_Data_AUs_multi_apex_softmax0 import VoxCeleb2 as VoxCeleb2

from net.resnet_multi_view import ResNet_GCN_two_views
from net.resnet_multi_view_kd_orth import ResNet_kd_two_views 


from net.resnet10 import resnet10
from net.resnet10 import resnet18
from net.resnet10 import resnet34
from kd_losses import *


#BASE_LOCATION = os.environ['BASE_LOCATION']
BASE_LOCATION = ' '
VOX_CELEB_LOCATION=" "

arguments = argparse.ArgumentParser()
arguments.add_argument('--lr', type=float, default=0.1)
arguments.add_argument('--momentum', type=float, default=0.9)
arguments.add_argument('--load_old_model', action='store_true', default=False)
arguments.add_argument('--num_views', type=int, default=2, help='Number of source views + 1 (e.g. the target view) so set = 2 for 1 source view')
arguments.add_argument('--continue_epoch', type=int, default=0)
arguments.add_argument('--crop_size', type=int, default=200)
arguments.add_argument('--num_additional_ids', type=int, default=32)
arguments.add_argument('--use_landmark_supervision', action='store_true', default=False)
arguments.add_argument('--use_landmark_mask_supervision', action='store_true', default=False)
arguments.add_argument('--num_workers', type=int, default=1)
arguments.add_argument('--max_percentile', type=float, default=0.85)
arguments.add_argument('--diff_percentile', type=float, default=0.1)
arguments.add_argument('--batch_size', type=int, default=56)
arguments.add_argument('--log_dir', type=str, default=BASE_LOCATION+'/code_faces/runs/')
arguments.add_argument('--embedding_size', type=int, default=256)
arguments.add_argument('--run_dir', type=str, default='curriculumwidervox2%d_fabnet%s/lr_%.4f_lambda%.4f_nv%d_addids%d_cropsize%d')
arguments.add_argument('--old_model', type=str, default=BASE_LOCATION + '')
arguments.add_argument('--model_epoch_path', type=str, default='c%.4f_emb%d_bs%d_lambda%.4f_photomask%s_nv%d_addids%d_cropsize%d')
arguments.add_argument('--learn_mask', action='store_true')
opt = arguments.parse_args()


opt.run_dir = opt.run_dir % (opt.embedding_size, str(opt.use_landmark_supervision), opt.lr, 0, opt.num_views, opt.num_additional_ids, opt.crop_size)
opt.model_epoch_path = opt.model_epoch_path % (opt.lr, opt.embedding_size, opt.batch_size, 0, str(opt.use_landmark_supervision), opt.num_views, opt.num_additional_ids, opt.crop_size)

opt.model_epoch_path = 'epoch%d.pth'


##################################
########## parameters ############
##################################
### fusion_mode: 0 no fusion / 1 concate / 2 mean / 3 concate two layer
### database: 0 EmotioNet / 1 BP4D
### use_web: 0 only test images/ 1 training with unlabeled images
### lambda_co_regularization:
### AU_idx: the AU idxes you want to consider
##################################
fusion_mode = 1;
database = 0;
use_web = 0;

AU_num = 12;
AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11];

lambda_co_regularization = 100;
lambda_multi_view = 400;
###################################


###################################
########## network ################
###################################
net = ResNet_GCN_two_views(AU_num=AU_num, AU_idx=AU_idx, output=1, fusion_mode=fusion_mode, database=database);
model_path = './model/EmotioNet_model.pth.tar';
temp = torch.load(model_path);
net.load_state_dict(temp['net'])
net.cuda();
net.eval();
net.train(mode=False)
####################################



model = ResNet_kd_two_views()
model.lr = opt.lr
model.momentum = opt.momentum
model = model.cuda()

criterion_reconstruction = nn.L1Loss(reduce=False).cuda()
criterion = nn.CrossEntropyLoss()

#opt.learning_rate*(0.1**(epoch//30))


if opt.num_views > 1:
	optimizer = optim.SGD([{'params' : model.parameters(), 'lr' : 0}], lr=opt.lr, momentum=opt.momentum)
       
else:
	optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
       




T=4.0
P=2.0
kd_mode='spat'
# define loss functions
if kd_mode == 'logits':
	criterionKD = Logits()
elif kd_mode == 'st':
	criterionKD = SoftTarget(T)
elif kd_mode == 'at':
	criterionKD = AT(P)
elif kd_mode == 'fitnet':
	criterionKD = Hint()
elif kd_mode == 'nst':
	criterionKD = NST()
elif kd_mode == 'pkt':
	criterionKD = PKTCosSim()
elif kd_mode == 'fsp':
	criterionKD = FSP()
elif kd_mode == 'rkd':
	criterionKD = RKD(args.w_dist, args.w_angle)
elif kd_mode == 'ab':
	criterionKD = AB(args.m)
elif kd_mode == 'sobolev':
	criterionKD = Sobolev()
elif kd_mode == 'cc':
	criterionKD = CC(args.gamma, args.P_order)
elif kd_mode == 'sp':
	criterionKD = SP()
elif kd_mode == 'spat':
	criterionKD = SPAT()






class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


criterion = nn.CrossEntropyLoss()
	
num_classes=2
def train(epoch, model, criterion, optimizer, num_additional_ids=5, minpercentile=0, maxpercentile=50):
	
        print(epoch)
        #print('*******************************************')
        opt.num_views=2
        train_set = VoxCeleb2(opt.num_views, epoch, 1, jittering=True)
        #print(train_set)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
        lr_dec=40
       
        optimizer = optim.SGD([{'params' : model.parameters(), 'lr':0.1*opt.lr*(0.1**(epoch//lr_dec))},
                                                
                                                        {'params' : net.net1.parameters(), 'lr':0.0},
							{'params' : net.net2.parameters(), 'lr' : 0.0*opt.lr*(0.1**(epoch//lr_dec))}
],  lr=opt.lr*(0.1**(epoch//lr_dec)),momentum=opt.momentum)
        
        t_loss = 0

       
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
                optimizer.zero_grad()
                   
                label=batch[1]    

                batch=batch[0]
         
                input_image = Variable(batch[1]).cuda()
                
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
             
                label1=label[0]
                label1 = label1.cuda()
                label1 = Variable(label1)

                label2=label[1]
                label2 = label2.cuda()
                label2 = Variable(label2)

                label3=label[2]
                label3 = label3.cuda()
                label3 = Variable(label3)

                label4=label[3]
                label4 = label4.cuda()
                label4 = Variable(label4)

                label5=label[4]
                label5 = label5.cuda()
                label5 = Variable(label5)

                label6=label[5]
                label6 = label6.cuda()
                label6 = Variable(label6)

                label7=label[6]
                label7= label7.cuda()
                label7 = Variable(label7)

                label8=label[7]
                label8 = label8.cuda()
                label8 = Variable(label8)
               



                ########Student model################
                output1,output2,output3,output4,output5,output6,output7,output8,Sconv1,Sconv2,Sconv3,Sconv4,Sconv5,Sfeat1,Sfc1,Sfc2, Sfc3, Sfc4,Sfc5,Sfc6,Sfc7,Sfc8,SSconv1,SSconv2,SSconv3,SSconv4,SSconv5,SSfeat2,SSfc1,SSfc2,SSfc3,SSfc4,SSfc5,SSfc6,SSfc7,SSfc8,loss_orth=model(input_image);
       


               
                #####################################
                ########Teacher model################
                #output1, conv1, conv2, conv3, conv4, conv5,feat1 = net(input_image);
                Toutput, Tconv1, Tconv2, Tconv3, Tconv4, Tconv5,Tfeat, TSoutput, TSconv1, TSconv2, TSconv3, TSconv4, TSconv5,TSfeat = net(input_image);




                loss1_temp1=criterion(Sfc1, label1).mean()
                loss1_temp2=criterion(Sfc2, label2).mean()
                loss1_temp3=criterion(Sfc3, label3).mean()
                loss1_temp4=criterion(Sfc4, label4).mean()
                loss1_temp5=criterion(Sfc5, label5).mean()
                loss1_temp6=criterion(Sfc6, label6).mean()
                loss1_temp7=criterion(Sfc7, label7).mean()
                loss1_temp8=criterion(Sfc8, label8).mean()



                loss2_temp1=criterion(SSfc1, label1).mean()
                loss2_temp2=criterion(SSfc2, label2).mean()
                loss2_temp3=criterion(SSfc3, label3).mean()
                loss2_temp4=criterion(SSfc4, label4).mean()
                loss2_temp5=criterion(SSfc5, label5).mean()
                loss2_temp6=criterion(SSfc6, label6).mean()
                loss2_temp7=criterion(SSfc7, label7).mean()
                loss2_temp8=criterion(SSfc8, label8).mean()

                
                
                fc1=(SSfc1+Sfc1)/2.0
                fc2=(SSfc2+Sfc2)/2.0
                fc3=(SSfc3+Sfc3)/2.0
                fc4=(SSfc4+Sfc4)/2.0
                fc5=(SSfc5+Sfc5)/2.0
                fc6=(SSfc6+Sfc6)/2.0
                fc7=(SSfc7+Sfc7)/2.0
                fc8=(SSfc8+Sfc8)/2.0

                
                loss_temp1=criterion(fc1, label1).mean()
                loss_temp2=criterion(fc2, label2).mean()
                loss_temp3=criterion(fc3, label3).mean()
                loss_temp4=criterion(fc4, label4).mean()
                loss_temp5=criterion(fc5, label5).mean()
                loss_temp6=criterion(fc6, label6).mean()
                loss_temp7=criterion(fc7, label7).mean()
                loss_temp8=criterion(fc8, label8).mean()

        

                if kd_mode in ['logits', 'st']:
                        kd_loss = criterionKD(Tfeat, Sfeat) 
                elif kd_mode in ['fitnet', 'nst']:
                        kd_loss1 = criterionKD(Sconv5, Tconv5) 
                        kd_loss2 = criterionKD(SSconv5, TSconv5) 
                elif kd_mode in ['at', 'sp','spat']:
                        kd_loss1 = (criterionKD(Sconv1, Tconv1) +
					   criterionKD(Sconv2, Tconv2) +
					   criterionKD(Sconv3, Tconv3)  +
					   criterionKD(Sconv4, Tconv4)  +
					   criterionKD(Sconv5, Tconv5))/ 5.0

                
                        kd_loss2 = (criterionKD(SSconv1, TSconv1) +
					   criterionKD(SSconv2, TSconv2) +
					   criterionKD(SSconv3, TSconv3)  +
					   criterionKD(SSconv4, TSconv4)  +
					   criterionKD(SSconv5, TSconv5))/ 5.0    

                elif kd_mode in ['ab']:
                        kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
					   criterionKD(rb2_s[0], rb2_t[0].detach()) +
					   criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 



                

                
                AU_loss1=(loss1_temp1+loss1_temp2+loss1_temp3+loss1_temp4+loss1_temp5+loss1_temp6+loss_temp7+loss1_temp8)/8
                AU_loss2=(loss2_temp1+loss2_temp2+loss2_temp3+loss2_temp4+loss2_temp5+loss2_temp6+loss2_temp7+loss2_temp8)/8
                AU_loss=(loss_temp1+loss_temp2+loss_temp3+loss_temp4+loss_temp5+loss_temp6+loss_temp7+loss_temp8)/8

               
                loss=AU_loss+AU_loss1+AU_loss2+100.0*kd_loss1*(1.0**(epoch//lr_dec))+100.0*kd_loss2*(1.0**(epoch//lr_dec))+100*loss_orth
              

                loss.backward()
        
                t_loss += loss.cpu().data
		
                optimizer.step()

                logits1 = F.softmax(fc1,dim=1)
                logits2 = F.softmax(fc2,dim=1)
                logits3 = F.softmax(fc3,dim=1)
                logits4 = F.softmax(fc4,dim=1)
                logits5 = F.softmax(fc5,dim=1)
                logits6 = F.softmax(fc6,dim=1)
                logits7 = F.softmax(fc7,dim=1)
                logits8 = F.softmax(fc8,dim=1)


              
                true_label1 =label1
                true_label2=label2
                true_label3=label3
                true_label4=label4
                true_label5=label5
                true_label6=label6
                true_label7=label7
                true_label8=label8


                _, pred1 = logits1.topk(1, 1, True)
                pred1 = pred1.t()[0]
               
                _, pred2 = logits2.topk(1, 1, True)
                pred2 = pred2.t()[0]
             
                _, pred3 = logits3.topk(1, 1, True)
                pred3 = pred3.t()[0]
              
                _, pred4 = logits4.topk(1, 1, True)
                pred4 = pred4.t()[0]
            
                _, pred5 = logits5.topk(1, 1, True)
                pred5 = pred5.t()[0]
               
                _, pred6 = logits6.topk(1, 1, True)
                pred6 = pred6.t()[0]
               
                _, pred7 = logits7.topk(1, 1, True)
                pred7 = pred7.t()[0]
              
                _, pred8 = logits8.topk(1, 1, True)
                pred8 = pred8.t()[0]
		

                f1score1=f1_score(true_label1.cpu().numpy(), pred1.cpu().numpy(),average='macro')         
                f1score2=f1_score(true_label2.cpu().numpy(), pred2.cpu().numpy(),average='macro') 
                f1score3=f1_score(true_label3.cpu().numpy(), pred3.cpu().numpy(),average='macro')
                f1score4=f1_score(true_label4.cpu().numpy(), pred4.cpu().numpy(),average='macro')
                f1score5=f1_score(true_label5.cpu().numpy(), pred5.cpu().numpy(),average='macro') 
                f1score6=f1_score(true_label6.cpu().numpy(), pred6.cpu().numpy(),average='macro')
                f1score7=f1_score(true_label7.cpu().numpy(), pred7.cpu().numpy(),average='macro')
                f1score8=f1_score(true_label8.cpu().numpy(), pred8.cpu().numpy(),average='macro') 
                f1score =(f1score1+f1score2+f1score3+f1score4+f1score5+f1score6+f1score7+f1score8)/8
                if epoch % 50 == 0:
                         print(f1score)
                        

                

                acc1 = accuracy_score(true_label1.cpu().numpy(),pred1.cpu().numpy())
                acc2 = accuracy_score(true_label2.cpu().numpy(),pred2.cpu().numpy())
                acc3 = accuracy_score(true_label3.cpu().numpy(),pred3.cpu().numpy())
                acc4 = accuracy_score(true_label4.cpu().numpy(),pred4.cpu().numpy())
                acc5 = accuracy_score(true_label5.cpu().numpy(),pred5.cpu().numpy())
                acc6 = accuracy_score(true_label6.cpu().numpy(),pred6.cpu().numpy())
                acc7 = accuracy_score(true_label7.cpu().numpy(),pred7.cpu().numpy())
                acc8 = accuracy_score(true_label8.cpu().numpy(),pred8.cpu().numpy())
                acc =(acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8)/8
                if epoch % 50 == 0:
                         print(acc)
                       




        print('training_loss',t_loss.cpu().data)
        return {'reconstruction_error' : t_loss / float(iteration)}



def val(epoch, model, criterion, optimizer, minpercentile=0, maxpercentile=50):
        val_set = VoxCeleb2(opt.num_views, 0, 2, jittering=True) 

        val_data_loader = DataLoader(dataset=val_set, num_workers=opt.num_workers, batch_size=1, shuffle=False)
    
        t_loss = 0
        targets_all=[]
        predict_all=[]
        targets_all1=[]
        predict_all1=[]
        targets_all2=[]
        predict_all2=[]
        targets_all3=[]
        predict_all3=[]
        targets_all4=[]
        predict_all4=[]
        targets_all5=[]
        predict_all5=[]
        targets_all6=[]
        predict_all6=[]
        targets_all7=[]
        predict_all7=[]
        targets_all8=[]
        predict_all8=[]
 
              
        
        for iteration, batch in enumerate(val_data_loader, 1):

                label=batch[1]
                batch=batch[0]
                input_image = Variable(batch[1]).cuda()
                         
                label1=label[0]
                label1 = label1.cuda()
                label1 = Variable(label1)
                
                label2=label[1]
                label2 = label2.cuda()
                label2 = Variable(label2)

                label3=label[2]
                label3 = label3.cuda()
                label3 = Variable(label3)

                label4=label[3]
                label4 = label4.cuda()
                label4 = Variable(label4)

                label5=label[4]
                label5 = label5.cuda()
                label5 = Variable(label5)

                label6=label[5]
                label6 = label6.cuda()
                label6 = Variable(label6)

                label7=label[6]
                label7= label7.cuda()
                label7 = Variable(label7)

                label8=label[7]
                label8 = label8.cuda()
                label8 = Variable(label8)

	
		
                ########Student model################
                output1,output2,output3,output4,output5,output6,output7,output8,Sconv1,Sconv2,Sconv3,Sconv4,Sconv5,Sfeat1,Sfc1,Sfc2, Sfc3, Sfc4,Sfc5,Sfc6,Sfc7,Sfc8,SSconv1,SSconv2,SSconv3,SSconv4,SSconv5,SSfeat2,SSfc1,SSfc2,SSfc3,SSfc4,SSfc5,SSfc6,SSfc7,SSfc8,loss_orth=model(input_image);
             

                fc1=(SSfc1+Sfc1)/2.0
                fc2=(SSfc2+Sfc2)/2.0
                fc3=(SSfc3+Sfc3)/2.0
                fc4=(SSfc4+Sfc4)/2.0
                fc5=(SSfc5+Sfc5)/2.0
                fc6=(SSfc6+Sfc6)/2.0
                fc7=(SSfc7+Sfc7)/2.0
                fc8=(SSfc8+Sfc8)/2.0



                
                logits1 = F.softmax(fc1,dim=1)
                logits2 = F.softmax(fc2,dim=1)
                logits3 = F.softmax(fc3,dim=1)
                logits4 = F.softmax(fc4,dim=1)
                logits5 = F.softmax(fc5,dim=1)
                logits6 = F.softmax(fc6,dim=1)
                logits7 = F.softmax(fc7,dim=1)
                logits8 = F.softmax(fc8,dim=1)

             
                true_label1=label1   
                true_label2=label2
                true_label3=label3
                true_label4=label4
                true_label5=label5             
                true_label6=label6 
                true_label7=label7
                true_label8=label8


                _, pred1 = logits1.topk(1, 1, True)
                pred1 = pred1.t()[0]
               
                _, pred2 = logits2.topk(1, 1, True)
                pred2 = pred2.t()[0]
             
                _, pred3 = logits3.topk(1, 1, True)
                pred3 = pred3.t()[0]
              
                _, pred4 = logits4.topk(1, 1, True)
                pred4 = pred4.t()[0]
            
                _, pred5 = logits5.topk(1, 1, True)
                pred5 = pred5.t()[0]
               
                _, pred6 = logits6.topk(1, 1, True)
                pred6 = pred6.t()[0]
               
                _, pred7 = logits7.topk(1, 1, True)
                pred7 = pred7.t()[0]
              
                _, pred8 = logits8.topk(1, 1, True)
                pred8 = pred8.t()[0]
               
                
                loss_temp1=criterion(fc1, label1).mean()
                loss_temp2=criterion(fc2, label2).mean()
                loss_temp3=criterion(fc3, label3).mean()
                loss_temp4=criterion(fc4, label4).mean()
                loss_temp5=criterion(fc5, label5).mean()
                loss_temp6=criterion(fc6, label6).mean()
                loss_temp7=criterion(fc7, label7).mean()
                loss_temp8=criterion(fc8, label8).mean()
                loss=(loss_temp1+loss_temp2+loss_temp3+loss_temp4+loss_temp5+loss_temp6+loss_temp7+loss_temp8)

                #print('validation_loss',loss)

                targets_all1=np.append(targets_all1,true_label1.cpu().numpy()) 
                predict_all1=np.append(predict_all1,pred1.cpu().numpy())
                targets_all2=np.append(targets_all2,true_label2.cpu().numpy()) 
                predict_all2=np.append(predict_all2,pred2.cpu().numpy())
                targets_all3=np.append(targets_all3,true_label3.cpu().numpy()) 
                predict_all3=np.append(predict_all3,pred3.cpu().numpy())
                targets_all4=np.append(targets_all4,true_label4.cpu().numpy()) 
                predict_all4=np.append(predict_all4,pred4.cpu().numpy())
                targets_all5=np.append(targets_all5,true_label5.cpu().numpy()) 
                predict_all5=np.append(predict_all5,pred5.cpu().numpy())
                targets_all6=np.append(targets_all6,true_label6.cpu().numpy()) 
                predict_all6=np.append(predict_all6,pred6.cpu().numpy())
                targets_all7=np.append(targets_all7,true_label7.cpu().numpy()) 
                predict_all7=np.append(predict_all7,pred7.cpu().numpy())
                targets_all8=np.append(targets_all8,true_label8.cpu().numpy()) 
                predict_all8=np.append(predict_all8,pred8.cpu().numpy())
      
       
                t_loss += loss.cpu().data

                
		
        f1score1sb=f1_score(targets_all1,predict_all1,average='binary')
        f1score2sb=f1_score(targets_all2,predict_all2,average='binary')
        f1score3sb=f1_score(targets_all3,predict_all3,average='binary')
    
        f1score5sb=f1_score(targets_all5,predict_all5,average='binary')
   
        f1score4sb=f1_score(targets_all4,predict_all4,average='binary')
        f1score6sb=f1_score(targets_all6,predict_all6,average='binary')
        f1score7sb=f1_score(targets_all7,predict_all7,average='binary')
        f1score8sb=f1_score(targets_all8,predict_all8,average='binary')

        f1score1s=f1_score(targets_all1,predict_all1,average='macro')
        f1score2s=f1_score(targets_all2,predict_all2,average='macro')
        f1score3s=f1_score(targets_all3,predict_all3,average='macro')
    
        f1score5s=f1_score(targets_all5,predict_all5,average='macro')
   
        f1score4s=f1_score(targets_all4,predict_all4,average='macro')
        f1score6s=f1_score(targets_all6,predict_all6,average='macro')
        f1score7s=f1_score(targets_all7,predict_all7,average='macro')
        f1score8s=f1_score(targets_all8,predict_all8,average='macro')
   
        acc1s=accuracy_score(targets_all1,predict_all1)
        acc2s=accuracy_score(targets_all2,predict_all2)
        acc3s=accuracy_score(targets_all3,predict_all3)
 
        acc5s=accuracy_score(targets_all5,predict_all5)
  	
        acc4s=accuracy_score(targets_all4,predict_all4)
        acc6s=accuracy_score(targets_all6,predict_all6)
        acc7s=accuracy_score(targets_all7,predict_all7)
        acc8s=accuracy_score(targets_all8,predict_all8)

        targets_all_final=np.concatenate((targets_all1, targets_all2,targets_all3, targets_all4,targets_all5, targets_all6,targets_all7,targets_all8),axis=0)
        #print(targets_all_final.shape)
        predict_all_final=np.concatenate((predict_all1, predict_all2,predict_all3, predict_all4,predict_all5, predict_all6,predict_all7,predict_all8),axis=0)
       
        target_name =('DVspat_at_casme2_18target1.mat')
        target_path = os.path.join('/research/tklab/personal/ytli/pytorch RES3D CVPR2018/FAb-Net-master/FAb-Net/AUs_xuesong/MLCR-master AU/result/', target_name)
        scipy.io.savemat(target_path, {'target':targets_all_final})
            #preds_db['all'].numpy()

        pred_name =('DVspat_at_casme2_18pred1.mat')
        pred_path = os.path.join('/research/tklab/personal/ytli/pytorch RES3D CVPR2018/FAb-Net-master/FAb-Net/AUs_xuesong/MLCR-master AU/result/', pred_name)
        scipy.io.savemat(pred_path, {'pred': predict_all_final })




        print('f1score_all_binary')
        print(f1score1sb,f1score2sb,f1score3sb,f1score4sb,f1score5sb,f1score6sb,f1score7sb,f1score8sb)
        f1scoresb=(f1score1sb+f1score2sb+f1score3sb+f1score4sb+f1score5sb+f1score6sb+f1score7sb+f1score8sb)/8
        print(f1scoresb)


        print('f1score_all_macro')
        print(f1score1s,f1score2s,f1score3s,f1score4s,f1score5s,f1score6s,f1score7s,f1score8s)
        f1scores=(f1score1s+f1score2s+f1score3s+f1score4s+f1score5s+f1score6s+f1score7s+f1score8s)/8
        print(f1scores)
  
        print('acc_all')
        print(acc1s,acc2s,acc3s,acc4s,acc5s,acc6s,acc7s,acc8s)
        accscores=(acc1s+acc2s+acc3s+acc4s+acc5s+acc6s+acc7s+acc8s)/8
        print(accscores)





        return {'reconstruction_error' : t_loss / float(iteration)} 

def checkpoint(model, save_path):
	checkpoint_state = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : model.epoch, 
							'lr' : model.lr, 'momentum' : model.momentum, 'opts' : opt}

	torch.save(checkpoint_state, save_path)

def run(minpercentile=0, maxpercentile=0.5):
        scheduler = TrackLoss()
 
        MODEL_PATH = '/research/tklab/personal/ytli/pytorch RES3D CVPR2018/FAb-Net-master/FAb-Net/AUs_xuesong/MLCR-master/model'
     
	#plateauscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        if opt.continue_epoch > 0:
                print('load_model')
                past_state = torch.load(opt.model_epoch_path % (opt.continue_epoch - 1))
                model.load_state_dict(torch.load(opt.model_epoch_path % (opt.continue_epoch - 1))['state_dict'])
                optimizer.load_state_dict(torch.load(opt.model_epoch_path % (opt.continue_epoch - 1))['optimizer'])

                percentiles = past_state['opts']
                minpercentile = percentiles.minpercentile
                maxpercentile = percentiles.maxpercentile     

        for epoch in range(opt.continue_epoch,101):
                model.epoch = epoch
                model.optimizer_state = optimizer.state_dict()
                model.train()

                train_loss = train(epoch, model,criterion, optimizer, minpercentile=minpercentile, maxpercentile=maxpercentile)
		

                if epoch==80:
    
                        print(epoch)
                        model.eval()
                      
                        loss = val(epoch, model,criterion, optimizer, minpercentile=0, maxpercentile=1)

                        checkpoint(model, opt.model_epoch_path % epoch)
                        torch.save(model, 'Twoview-kd-model.pt')
			

                opt.minpercentile = minpercentile
                opt.maxpercentile = maxpercentile



if __name__ == '__main__':
         if opt.load_old_model:
                model.load_state_dict(torch.load(opt.old_model)['state_dict'])
                percentiles = torch.load(opt.old_model)['opts']
                minpercentile = percentiles.minpercentile
                maxpercentile = percentiles.maxpercentile

                run(minpercentile, maxpercentile)
         else:
                run()





