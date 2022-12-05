import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            #nn.ReLU(inplace=True),
           # nn.Dropout(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
            #nn.Linear(4096, num_classes),
        )

        feat_len=4096
        self.fc_1 = nn.Linear(feat_len, 2)
	self.fc_2 = nn.Linear(feat_len, 2)
	self.fc_3 = nn.Linear(feat_len, 2)
	self.fc_4 = nn.Linear(feat_len, 2)
	self.fc_5 = nn.Linear(feat_len, 2)
	self.fc_6 = nn.Linear(feat_len, 2)
	self.fc_7 = nn.Linear(feat_len, 2)
	self.fc_8 = nn.Linear(feat_len, 2)

        self.fc1 = nn.Linear(feat_len, 2)
	self.fc2 = nn.Linear(feat_len, 2)
	self.fc3 = nn.Linear(feat_len, 2)
	self.fc4 = nn.Linear(feat_len, 2)
	self.fc5 = nn.Linear(feat_len, 2)
	self.fc6 = nn.Linear(feat_len, 2)
	self.fc7 = nn.Linear(feat_len, 2)
	self.fc8 = nn.Linear(feat_len, 2)

     
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))

    MODEL_PATH = '/research/tklab/personal/ytli/pytorch RES3D CVPR2018/FAb-Net-master/FAb-Net/pytorch-playground-master'
        
    classifier_model = MODEL_PATH + '/alexnet-owt-4df8aa71.pth'   
      
    # Load the model and classifier weights
    pretrain=torch.load(classifier_model)
    model_dict =  model.state_dict()
    state_dict = {k:v for k,v in pretrain.items() if k in  model_dict.keys()}
    print(state_dict.keys())  
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    return model
