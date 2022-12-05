import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import Function
import random
import math

import sys
sys.path.append('.');
from loss.loss_BCE import BCE_sigmoid_negtive_bias_all;

class MultiView_all_loss(nn.Module):
    def __init__(self, AU_num = 5, AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11], fusion_mode = 0,
                 use_web = 0, database = 0, lambda_co_regularization = 400, lambda_multi_view = 100):

        super(MultiView_all_loss, self).__init__()

        self.lossfunc = BCE_sigmoid_negtive_bias_all(size_average=True, AU_num = AU_num, AU_idx = AU_idx, database = database);

        self.BCE = nn.BCELoss();
        self.sigmoid = nn.Sigmoid();
        self.log_sigmoid = nn.LogSigmoid();

        self.AU_num = AU_num;
        self.AU_idx = AU_idx;
        self.lambda_co_regularization = lambda_co_regularization;
        self.lambda_multi_view = lambda_multi_view;
        self.eps = 0.001;

        self.fusion_mode = fusion_mode;
        self.use_web = use_web;

        
        self.loss=nn.CrossEntropyLoss()

    def forward(self, gt, pre, pre1, pre2, weight1, bias1, weight2, bias2, feat1, feat2, flag):

        N = gt.size(0);

        mask = flag.eq(1);
        '''
        pre_label1 = torch.masked_select(pre1, mask);
        pre_label1 = pre_label1.view(-1, self.AU_num);

        pre_label2 = torch.masked_select(pre2, mask);
        pre_label2 = pre_label2.view(-1, self.AU_num);

        pre_label = torch.masked_select(pre, mask);
        pre_label = pre_label.view(-1, self.AU_num);

        gt = torch.masked_select(gt, mask);
        gt = gt.view(-1, self.AU_num);
        '''



        if bool(gt.numel()):
            loss_pred = self.lossfunc(pre_label, gt);
            loss_pred1 = self.lossfunc(pre_label1, gt);
            loss_pred2 = self.lossfunc(pre_label2, gt);
        else:
            loss_pred = Variable(torch.FloatTensor([0])).cuda();
            loss_pred1 = Variable(torch.FloatTensor([0])).cuda();
            loss_pred2 = Variable(torch.FloatTensor([0])).cuda();

        if self.fusion_mode == 0:
            loss_BCE = (loss_pred1 + loss_pred2)/2;
        else:
            loss_BCE = loss_pred + (loss_pred1 + loss_pred2)/2;

        ############### loss multi-view ########
        loss_multi_view = torch.FloatTensor([0]);
        loss_multi_view = loss_multi_view.cuda();

        bias1 = bias1.view(self.AU_num, -1);
        feat1 = torch.cat((weight1, bias1), 1);
        bias2 = bias2.view(self.AU_num, -1);
        feat2 = torch.cat((weight2, bias2), 1);

        tmp = torch.norm(feat1, 2, 1);
        feat_norm1 = feat1 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(feat2, 2, 1);
        feat_norm2 = feat2 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth = torch.mean(torch.abs(x));
        loss_multi_view = loss_multi_view + loss_weight_orth;

        loss_multi_view = loss_multi_view*self.lambda_multi_view;
        ############ end loss multi-view #######

        ################# J-S divergence #################
        loss_similar = torch.FloatTensor([0]);
        loss_similar = loss_similar.cuda();

        

        loss_similar = loss_similar * self.lambda_co_regularization;
        ################# end J-S divergence #################

        loss = loss_BCE + loss_multi_view + loss_similar;

        return loss, loss_pred, loss_pred1, loss_pred2, loss_multi_view, loss_similar
