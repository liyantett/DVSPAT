import torch
import torch.nn as nn

from net.resnet import resnet34;




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





class ResNet_GCN_two_views(nn.Module):
    def __init__(self, AU_num = 12, AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11], output = 1, fusion_mode = 0, database = 0):
        super(ResNet_GCN_two_views, self).__init__()
        
        self.net1 = resnet34(num_classes = AU_num, num_output=66);
        self.net2 = resnet34(num_classes = AU_num, num_output=66);

        self.AU_num = AU_num;
        self.AU_idx = AU_idx;
        self.fusion_mode = fusion_mode;
        self.output = output;
        self.scale = 1;

        # Different methods to fuse the features of the two views.
        # For fair comparasion, we choose to not fuse the two features.
        if self.fusion_mode == 0 or self.fusion_mode == 1:
            self.fc = nn.Linear(1024, AU_num);
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
        self.relation = GCN_layer(num_classes = AU_num, AU_idx = self.AU_idx, in_channel = 513, database = database);

    def forward(self, data):

        N = data.size(0);
       
        output1, conv1, conv2, conv3, conv4, conv5, feat1 = self.net1(data);
        output2, conv12, conv22, conv32, conv42, conv52,feat2 = self.net2(data);
        
        weight1 = self.net1.fc.weight;
        bias1 = self.net1.fc.bias;
        weight2 = self.net2.fc.weight;
        bias2 = self.net2.fc.bias;

        bias1 = bias1.view(self.AU_num, -1);
        weight_norm1 = torch.cat((weight1, bias1), 1);
        bias2 = bias2.view(self.AU_num, -1);
        weight_norm2 = torch.cat((weight2, bias2), 1);

        feat_norm1 = feat1;
        feat_norm2 = feat2;

        if self.Is_begin_weight:
            self.begin_weight1 = weight_norm1;
            
            self.begin_weight2 = weight_norm2;
            self.Is_begin_weight = False;
        else:
            weight_norm1 = self.relation(self.begin_weight1.t());
            weight_norm1 = weight_norm1.t();
            weight_norm2 = self.relation(self.begin_weight2.t());
            weight_norm2 = weight_norm2.t();

        #print(feat_norm1.shape,weight_norm1.shape)
        output1 = torch.mm(feat_norm1, torch.t(weight_norm1[:, 0:512])) + weight_norm1[:, 512];
        #print(output1.shape)
        output1 = self.scale * output1;
        output2 = torch.mm(feat_norm2, torch.t(weight_norm2[:, 0:512])) + weight_norm2[:, 512];
        output2 = self.scale * output2;

        if self.fusion_mode == 0 or self.fusion_mode == 1:
            temp = torch.cat((feat1, feat2), 1);
            output = self.fc(temp);
        elif self.fusion_mode == 2:
            temp = (feat1 + feat2) / 2;
            output = self.fc(temp);
        elif self.fusion_mode == 3:
            temp = torch.cat((feat1, feat2), 1);
            output = self.fc(temp);
      
        
        return output1, conv1, conv2, conv3, conv4, conv5,feat1,output2, conv12, conv22, conv32, conv42, conv52,feat2
       
