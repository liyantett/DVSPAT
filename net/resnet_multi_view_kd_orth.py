import torch
import torch.nn as nn

#from net.resnet import resnet34;
from net.resnet_kd import resnet18
from net.resnet_kd import resnet34
from net.resnet_gcn import GCN_layer;



def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 4):
        ft_module_names.append('layer{}'.format(i))


    #ft_module_names.append('tanh')
    #ft_module_names.append('layer_reduce_bn')
    #ft_module_names.append('layer_reduce_relu')
    ft_module_names.append('fc')
    ft_module_names.append('fc1')
    ft_module_names.append('fc2')
    ft_module_names.append('fc3')
    ft_module_names.append('fc4')
    ft_module_names.append('fc5')
    ft_module_names.append('fc6')
    ft_module_names.append('fc7')
    ft_module_names.append('fc2')
    ft_module_names.append('fc8')
    ft_module_names.append('fc9')
    ft_module_names.append('fc10')
    ft_module_names.append('fc11')
    print('fc++++++++++++++++++++++++++++++++++++++')
    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                print(ft_module)
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0001})

    return parameters





class ResNet_kd_two_views(nn.Module):
    def __init__(self, AU_num = 2, AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11], output = 1, fusion_mode = 0, database = 0):
        super(ResNet_kd_two_views, self).__init__()
        
        self.kd_net1 = resnet18();
        self.kd_net2 = resnet18();

        self.AU_num = AU_num;
        self.AU_idx = AU_idx;
        self.fusion_mode = fusion_mode;
        self.output = output;
        self.scale = 1;

        # Different methods to fuse the features of the two views.
        # For fair comparasion, we choose to not fuse the two features.
        dimfc=1024
        if self.fusion_mode == 0 or self.fusion_mode == 1:
            
            self.fc_1 = nn.Linear(dimfc, AU_num)
            self.fc_2 = nn.Linear(dimfc, AU_num)
            self.fc_3 = nn.Linear(dimfc, AU_num)
            self.fc_4 = nn.Linear(dimfc, AU_num)
            self.fc_5 = nn.Linear(dimfc, AU_num)
            self.fc_6 = nn.Linear(dimfc, AU_num)
            self.fc_7 = nn.Linear(dimfc, AU_num)
            self.fc_8 = nn.Linear(dimfc, AU_num)
            #self.fc_emotion = nn.Linear(1024, 5)
            
        elif self.fusion_mode == 2:
            self.fc = nn.Linear(512, AU_num);
        elif self.fusion_mode == 3:
            self.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ELU(),
                nn.Linear(512, AU_num)
            )

        self.Is_begin_weight = True;
        self.begin_weight1 = None;
        self.begin_weight2 = None;
        #self.relation = GCN_layer(num_classes = AU_num, AU_idx = self.AU_idx, in_channel = 513, database = database);

    def forward(self, data):

        N = data.size(0);
       
        output1, conv1, conv2, conv3, conv4, conv5, feat1,fc1,fc2,fc3,fc4,fc5,fc6,fc7,fc8,fc_emotion= self.kd_net1(data);
        output2, conv10, conv20, conv30, conv40, conv50,feat2,fc10,fc20,fc30,fc40,fc50,fc60,fc70,fc80,fc_emotion0= self.kd_net2(data);
        
        #au1
        weight1_1 = self.kd_net1.fc1.weight;
        bias1_1 = self.kd_net1.fc1.bias;
        weight2_1 = self.kd_net2.fc1.weight;
        bias2_1 = self.kd_net2.fc1.bias;

        bias1_1 = bias1_1.view(self.AU_num, -1);
        weight_norm1_1 = torch.cat((weight1_1, bias1_1), 1);
        bias2_1 = bias2_1.view(self.AU_num, -1);
        weight_norm2_1 = torch.cat((weight2_1, bias2_1), 1);

        #au2
        weight1_2 = self.kd_net1.fc2.weight;
        bias1_2 = self.kd_net1.fc2.bias;
        weight2_2 = self.kd_net2.fc2.weight;
        bias2_2 = self.kd_net2.fc2.bias;

        bias1_2 = bias1_2.view(self.AU_num, -1);
        weight_norm1_2 = torch.cat((weight1_2, bias1_2), 1);
        bias2_2 = bias2_2.view(self.AU_num, -1);
        weight_norm2_2 = torch.cat((weight2_2, bias2_2), 1);


        #au3
        weight1_3 = self.kd_net1.fc3.weight;
        bias1_3 = self.kd_net1.fc3.bias;
        weight2_3 = self.kd_net2.fc3.weight;
        bias2_3 = self.kd_net2.fc3.bias;

        
        bias1_3 = bias1_3.view(self.AU_num, -1);
        weight_norm1_3 = torch.cat((weight1_3, bias1_3), 1);
        bias2_3 = bias2_3.view(self.AU_num, -1);
        weight_norm2_3 = torch.cat((weight2_3, bias2_3), 1);

        #au4
        weight1_4 = self.kd_net1.fc4.weight;
        bias1_4 = self.kd_net1.fc4.bias;
        weight2_4 = self.kd_net2.fc4.weight;
        bias2_4 = self.kd_net2.fc4.bias;

        bias1_4 = bias1_1.view(self.AU_num, -1);
        weight_norm1_4 = torch.cat((weight1_4, bias1_4), 1);
        bias2_4 = bias2_1.view(self.AU_num, -1);
        weight_norm2_4 = torch.cat((weight2_4, bias2_4), 1);

        #au5
        weight1_5 = self.kd_net1.fc5.weight;
        bias1_5 = self.kd_net1.fc5.bias;
        weight2_5 = self.kd_net2.fc5.weight;
        bias2_5 = self.kd_net2.fc5.bias;

        bias1_5 = bias1_1.view(self.AU_num, -1);
        weight_norm1_5 = torch.cat((weight1_5, bias1_5), 1);
        bias2_5 = bias2_5.view(self.AU_num, -1);
        weight_norm2_5 = torch.cat((weight2_5, bias2_5), 1);

        #au6
        weight1_6 = self.kd_net1.fc6.weight;
        bias1_6 = self.kd_net1.fc6.bias;
        weight2_6 = self.kd_net2.fc6.weight;
        bias2_6 = self.kd_net2.fc6.bias;

        bias1_6 = bias1_6.view(self.AU_num, -1);
        weight_norm1_6 = torch.cat((weight1_6, bias1_6), 1);
        bias2_6 = bias2_6.view(self.AU_num, -1);
        weight_norm2_6 = torch.cat((weight2_6, bias2_6), 1);



        #au7
        weight1_7 = self.kd_net1.fc7.weight;
        bias1_7 = self.kd_net1.fc7.bias;
        weight2_7 = self.kd_net2.fc7.weight;
        bias2_7 = self.kd_net2.fc7.bias;

        bias1_7 = bias1_7.view(self.AU_num, -1);
        weight_norm1_7 = torch.cat((weight1_7, bias1_7), 1);
        bias2_7 = bias2_1.view(self.AU_num, -1);
        weight_norm2_7 = torch.cat((weight2_7, bias2_7), 1);


        #au8
        weight1_8 = self.kd_net1.fc8.weight;
        bias1_8 = self.kd_net1.fc8.bias;
        weight2_8 = self.kd_net2.fc8.weight;
        bias2_8 = self.kd_net2.fc8.bias;

        bias1_8 = bias1_8.view(self.AU_num, -1);
        weight_norm1_8 = torch.cat((weight1_8, bias1_8), 1);
        bias2_8 = bias2_8.view(self.AU_num, -1);
        weight_norm2_8 = torch.cat((weight2_8, bias2_8), 1);



        #emotion
        weight1_emotion = self.kd_net1.fc_emotion.weight;
        bias1_emotion = self.kd_net1.fc_emotion.bias;
        weight2_emotion = self.kd_net2.fc_emotion.weight;
        bias2_emotion = self.kd_net2.fc_emotion.bias;

        bias1_emotion = bias1_emotion.view(5, -1);
        weight_norm1_emotion = torch.cat((weight1_emotion, bias1_emotion), 1);
        bias2_emotion = bias2_emotion.view(5, -1);
        weight_norm2_emotion = torch.cat((weight2_emotion, bias2_emotion), 1);



        # loss 1 #############################################
        tmp = torch.norm(weight_norm1_1, 2, 1);
        feat_norm1 = weight_norm1_1 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(weight_norm2_1, 2, 1);
        feat_norm2 = weight_norm2_1 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth1 = torch.mean(torch.abs(x));
        # loss 2 #############################################
        tmp = torch.norm(weight_norm1_2, 2, 1);
        feat_norm1 = weight_norm1_2 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(weight_norm2_2, 2, 1);
        feat_norm2 = weight_norm2_2 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth2 = torch.mean(torch.abs(x));  
        # loss 3 #############################################
        tmp = torch.norm(weight_norm1_3, 2, 1);
        feat_norm1 = weight_norm1_3 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(weight_norm2_3, 2, 1);
        feat_norm2 = weight_norm2_3 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth3 = torch.mean(torch.abs(x));  
        # loss 4 #############################################
        tmp = torch.norm(weight_norm1_4, 2, 1);
        feat_norm1 = weight_norm1_4 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(weight_norm2_4, 2, 1);
        feat_norm2 = weight_norm2_4 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth4 = torch.mean(torch.abs(x));  


        # loss 5 #############################################
        tmp = torch.norm(weight_norm1_5, 2, 1);
        feat_norm1 = weight_norm1_5 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(weight_norm2_5, 2, 1);
        feat_norm2 = weight_norm2_5 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth5 = torch.mean(torch.abs(x));  

        # loss 6 #############################################
        tmp = torch.norm(weight_norm1_6, 2, 1);
        feat_norm1 = weight_norm1_6 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(weight_norm2_6, 2, 1);
        feat_norm2 = weight_norm2_6 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth6 = torch.mean(torch.abs(x));  

        # loss 7 #############################################
        tmp = torch.norm(weight_norm1_7, 2, 1);
        feat_norm1 = weight_norm1_7 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(weight_norm2_7, 2, 1);
        feat_norm2 = weight_norm2_7 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth7 = torch.mean(torch.abs(x));  


        # loss 8 #############################################
        tmp = torch.norm(weight_norm1_8, 2, 1);
        feat_norm1 = weight_norm1_8 / tmp.view(self.AU_num, -1);
        tmp = torch.norm(weight_norm2_8, 2, 1);
        feat_norm2 = weight_norm2_8 / tmp.view(self.AU_num, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orth8 = torch.mean(torch.abs(x));  




        # emotion #############################################
        tmp = torch.norm(weight_norm1_emotion, 2, 1);
        feat_norm1 = weight_norm1_emotion / tmp.view(5, -1);
        tmp = torch.norm(weight_norm2_emotion, 2, 1);
        feat_norm2 = weight_norm2_emotion / tmp.view(5, -1);

        x = feat_norm1 * feat_norm2;
        x = torch.sum(x, 1);
        loss_weight_orthemotion = torch.mean(torch.abs(x));  


 


        loss_weight_orth= (loss_weight_orth1+loss_weight_orth2+loss_weight_orth3+loss_weight_orth4+loss_weight_orth5+loss_weight_orth6+ loss_weight_orth7+loss_weight_orth8)/8 + loss_weight_orthemotion


        if self.fusion_mode == 0 or self.fusion_mode == 1:
            temp = torch.cat((feat1, feat2), 1);
            #temp = (feat1 + feat2) / 2;
            output1= self.fc_1(temp);
            output2= self.fc_2(temp);
            output3= self.fc_3(temp);
            output4= self.fc_4(temp);
            output5= self.fc_5(temp);
            output6= self.fc_6(temp);
            output7= self.fc_7(temp);
            output8= self.fc_8(temp);
            #output_emotion= self.fc_emotion(temp);


        elif self.fusion_mode == 2:
            temp = (feat1 + feat2) / 2;
            output = self.fc(temp);
        elif self.fusion_mode == 3:
            temp = torch.cat((feat1, feat2), 1);
            output = self.fc(temp);
      
        
        return output1,output2, output3, output4,output5,output6,output7,output8,conv1, conv2, conv3, conv4, conv5, feat1,fc1,fc2,fc3,fc4,fc5,fc6,fc7,fc8,  conv10, conv20, conv30, conv40, conv50,feat2,fc10,fc20,fc30,fc40,fc50,fc60,fc70,fc80, loss_weight_orth #, fc_emotion,fc_emotion0
       
