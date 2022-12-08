# Load the datase

import torch.utils.data as data
import numpy as np
from torchvision.transforms import ToTensor, Scale, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Resize,RandomRotation
from PIL import Image
import pandas as pd
import torch
import os


def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    return img

def load_img_mask(file_path):
	img = Image.open(file_path).convert('RGB')
	return img



VOX_CELEB_LOCATION=""

csv_path_train=''

csv_path_test=''



def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    file1 = open(csv_path, 'r')
    lines = file1.readlines()  
    keys = []
    key_labels = []
    key_labels_multi = []
    #for i in range(data.shape[0]):
    for line in lines:
        row = line.split(' ')
 
        basename = row[1]
    
        class_name1= int(row[3])
    
        class_name2= int(row[4])
   
        class_name3 = int(row[5])
         
        class_name4 =int(row[7])
       
        class_name5= int(row[10])
       
        class_name6 = int(row[11])
    
        class_name7 = int(row[12])

        class_name8 = int(row[13])
       
        class_name=[class_name1,class_name2,class_name3,class_name4,class_name5,class_name6,class_name7,
		class_name8]
        basename = VOX_CELEB_LOCATION+basename[70:]
        
        keys.append(basename)
        key_labels.append(class_name6)
        key_labels_multi.append(class_name)


    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        label_all = key_labels_multi[i]  
        database[key]['annotations'] = {'label': label}
        database[key]['annotations_all'] = {'label_all': label_all}
    return database,keys




class VoxCeleb2(data.Dataset):
	def __init__(self, num_views, random_seed, dataset, additional_face=True, jittering=False):
                if dataset == 1:                   
                        subset='training'
                        database,keys=convert_csv_to_dict(csv_path_train, subset)      #csv_path_test
                        self.ids = keys      #[0:120]  [121:237]
                        #self.ids = keys[0:120]
                        self.database=database
                        #print(self.ids)
                        #print(len(self.ids))
                if dataset == 2:
                        subset='validation'
                        database,keys=convert_csv_to_dict(csv_path_test, subset)     #csv_path_train
                        self.ids = keys     #[0:120]  [121:237]
                        self.database=database
                   
                if dataset == 3:
                        self.ids = np.load('../../Datasets/large_voxceleb/test.npy')
                self.rng = np.random.RandomState(random_seed)	
                self.num_views = num_views
                self.base_file = VOX_CELEB_LOCATION+ '/%s/' 
                crop = 210
                if jittering == True:
                    precrop = crop + 10
                    crop = self.rng.randint(crop, precrop)
                    self.pose_transform = Compose([Scale((224,224)),
                                               #Pad((20,80,20,30)),
                                               CenterCrop(precrop), RandomCrop(crop),
                                               Scale((224,224)), ToTensor()])
                    self.transform = Compose([Scale((224,224)),
                                          #Pad((20,80,20,30)),
                                          CenterCrop(precrop), RandomCrop(crop),RandomHorizontalFlip(p=0.5), 
                                          Scale((224,224)), ToTensor()])
                else:
                    precrop = crop
                    self.pose_transform = Compose([Scale((224,224)),
                                               #Pad((20,80,20,30)),
                                               CenterCrop(precrop),
                                               Scale((224,224)), ToTensor()])
                    self.transform = Compose([Scale((224,224)),
                                          #Pad((20,80,20,30)),
                                          CenterCrop(precrop),
                                          Scale((224,224)), ToTensor()])
	
	def __len__(self):
		#return self.ids.shape[0] - 1
		return len(self.ids)
	def __getitem__(self, index):
		#(other_face, _) = self.get_blw_item(self.rng.randint(self.__len__()))
		return self.get_blw_item(index)
	
	def get_blw_item(self, index):
		# Load the imag
                
                imgs = [0] * (self.num_views+1)
       
                img_track=self.ids
		 
                img_track_t=img_track
      
                img_track = img_track_t[self.rng.randint(len(img_track_t))]
      
                img_faces = [d for d in os.listdir(self.ids[index]+ '/' )]
      
                img_faces.sort()

                if self.num_views > len(img_faces):
                        img_index = self.rng.choice(range(len(img_faces)), self.num_views, replace=True)
                else:   
                        if self.database[self.ids[index] ]['subset'] =='training':
                                img_index_temp = self.rng.choice(range(3,7), self.num_views, replace=False)
                                img_index1 = img_index_temp[0]
                                img_index2 = img_index_temp[1]
                                img_index0 = self.rng.choice(range(0,1), self.num_views-1, replace=False)
                                img_index0 = img_index0[0]
                                img_index = [1,img_index1,img_index2] 
                                #print(img_index)
                        else:
                                img_index = [1,5,6]


                for i in range(0, 3):
                        img_name = self.ids[index] + '/' + img_faces[img_index[i]]
                        imgs[i] = load_img(img_name)
              
                        imgs[i] = self.transform(imgs[i])
 
                        image_label_AUs=self.database[self.ids[index]]['annotations_all']['label_all']
                label=image_label_AUs
                return imgs,label

