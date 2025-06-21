import os
import pandas as pd
from PIL import Image
from typing import Dict
import torch
from torch.utils.data import Dataset
import torchvision


class AmazonImageData(Dataset):
    
    def __init__(self, root:str, img_root:str, input_tokens:Dict, output_tokens:Dict, max_seq_len:int, transform:torchvision.transforms.Compose=None):
        super(AmazonImageData, self).__init__()
        self.img_root = img_root
        self.transform = transform
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.max_seq_len = max_seq_len
        self.file = pd.read_csv(root)
        self.file.iloc[:, 0] = self.file.iloc[:, 0].apply(lambda x: x[36:] if isinstance(x, str) else x)
        self.length = len(self.file)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.item()
            
        image = Image.open(self.img_root + '/' + self.file['image_link'][idx])
        if self.transform:
            image = self.transform(image)
        
        input_token = self.input_tokens[self.file['entity_name'][idx]]
        output_text = self.file['entity_value'][idx]
        entities = output_text.split(' ')
        output_tokens = [self.output_tokens['<sos>']]
        i = 0
        while i < len(entities):
            if entities[i] in self.output_tokens:
                output_tokens.append(self.output_tokens[entities[i]])
            elif entities[i] + ' ' + entities[i+1] in self.output_tokens:
                output_tokens.append(self.output_tokens[entities[i] + ' ' + entities[i+1]])
                i+=1
            else:
                for j in entities[i]:
                    output_tokens.append(self.output_tokens[j])
            output_tokens.append(self.output_tokens[' '])
            i+=1
        output_tokens.pop(-1)
        output_tokens.append(self.output_tokens['<eos>'])
        output_tokens.extend([self.output_tokens['<pad>']]*(self.max_seq_len-len(output_tokens)))
        output_tokens = torch.tensor(output_tokens, dtype=torch.int)

        return (image, input_token, output_tokens)