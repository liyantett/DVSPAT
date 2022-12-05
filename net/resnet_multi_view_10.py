import torch
import torch.nn as nn

from net.resnet10 import resnet10;
from net.resnet_gcn import GCN_layer;







class ResNet_kd_two_views(nn.Module):
    def __init__(self, class_num = 5, AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11], output = 1, fusion_mode = 0, database = 0):
        super(ResNet_kd_two_views, self).__init__()

        self.net1 = resnet10();
        self.net2 = resnet10();
        
        self.AU_num = 8;
        self.AU_idx = AU_idx;
        self.class_num=class_num
        self.fusion_mode = fusion_mode;
        self.output = output;
        self.scale = 1;





        # Different methods to fuse the features of the two views.
        # For fair comparasion, we choose to not fuse the two features.
        if self.fusion_mode == 0 or self.fusion_mode == 1:
            self.fc_emotion = nn.Linear(1024, class_num);
        elif self.fusion_mode == 2:
            self.fc_emotion = nn.Linear(512, class_num);
        elif self.fusion_mode == 3:
            self.fc_emotion = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ELU(),
                nn.Linear(512, class_num)
            )

        self.Is_begin_weight = True;
        self.begin_weight1 = None;
        self.begin_weight2 = None;
        #self.relation = GCN_layer(num_classes = AU_num, AU_idx = self.AU_idx, in_channel = 513, database = database);

    def forward(self, data):

        N = data.size(0);

        x, conv0s1, conv2s1, conv3s1, conv4s1, conv5s1, feats1 = self.net1(data);
        x, conv0s2, conv2s2, conv3s2, conv4s2, conv5s2, feats2= self.net2(data);
       
        weight1 = self.net1.fc_emotion.weight;
        bias1 = self.net1.fc_emotion.bias;
        weight2 = self.net2.fc_emotion.weight;
        bias2 = self.net2.fc_emotion.bias;

        bias1 = bias1.view(self.class_num, -1);
        weight_norm1 = torch.cat((weight1, bias1), 1);
        bias2 = bias2.view(self.class_num, -1);
        weight_norm2 = torch.cat((weight2, bias2), 1);

        
   
        if self.fusion_mode == 0 or self.fusion_mode == 1:
            temp = torch.cat((feat1, feat2), 1);
            output = self.fc_emotion(temp);
        elif self.fusion_mode == 2:
            temp = (feat1 + feat2) / 2;
            output = self.fc_emotion(temp);
        elif self.fusion_mode == 3:
            temp = torch.cat((feat1, feat2), 1);
            output = self.fc_emotion(temp);

        if self.output == 1:
            return weight1, bias1, weight2, bias2, feat1, feat2, output1, output2, output
        else:
            return output1, output2, output
