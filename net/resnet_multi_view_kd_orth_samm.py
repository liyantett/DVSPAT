import torch
import torch.nn as nn

#from net.resnet import resnet34;
from net.resnet_kd_samm import resnet18
from net.resnet_kd_samm import resnet34
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
        
        self.kd_net1 = resnet34();
        self.kd_net2 = resnet34();

        self.AU_num = AU_num;
        self.AU_idx = AU_idx;
        self.fusion_mode = fusion_mode;
        self.output = output;
        self.scale = 1;

        # Different methods to fuse the features of the two views.
        # For fair comparasion, we choose to not fuse the two features.
        if self.fusion_mode == 0 or self.fusion_mode == 1:
            
            self.fc_1 = nn.Linear(1024, AU_num)
	    self.fc_2 = nn.Linear(1024, AU_num)
	    self.fc_3 = nn.Linear(1024, AU_num)
	    self.fc_4 = nn.Linear(1024, AU_num)
	    

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
       
        output1, conv1, conv2, conv3, conv4, conv5, feat1,fc1,fc2,fc3,fc4= self.kd_net1(data);
        output2, conv10, conv20, conv30, conv40, conv50,feat2,fc10,fc20,fc30,fc40= self.kd_net2(data);
        
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


       
       
       
       


        loss_weight_orth= (loss_weight_orth1+loss_weight_orth2+loss_weight_orth3+loss_weight_orth4)/4


        if self.fusion_mode == 0 or self.fusion_mode == 1:
            temp = torch.cat((feat1, feat2), 1);
            output1= self.fc_1(temp);
            output2= self.fc_2(temp);
            output3= self.fc_3(temp);
            output4= self.fc_4(temp);
         


        elif self.fusion_mode == 2:
            temp = (feat1 + feat2) / 2;
            output = self.fc(temp);
        elif self.fusion_mode == 3:
            temp = torch.cat((feat1, feat2), 1);
            output = self.fc(temp);
      
        
        return output1,output2, output3, output4,conv1, conv2, conv3, conv4, conv5, feat1,fc1,fc2,fc3,fc4, conv10, conv20, conv30, conv40, conv50,feat2,fc10,fc20,fc30,fc40,loss_weight_orth
       
