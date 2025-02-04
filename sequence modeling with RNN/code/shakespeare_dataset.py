# -*- coding: utf-8 -*-

import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from torchvision import transforms


class RomeoDataset(Dataset):
    
    def __init__(self, filepath, crop_len, transform=None):
        
        ### Load data
        text = open(filepath, 'r').read()
        
        ### Preprocess data
        # Remove spaces after a new line
        text = re.sub('\n[ ]+', '\n', text)
        # Lower case
        text = text.lower()
        # Remove punctuation / rare symbols
        for c in ('"', '(', ')', '-', ':', ';', '1', '2', '3', '[', ']'):
            text = text.replace(c, "")
        # Extract the paragraph (divided by empty lines)
        par_list = re.split('\n\n', text)
        # Remove double new lines
        par_list = list(map(lambda s: s.replace('\n\n', '\n'), par_list))
        par_list = [x for x in par_list if len(x) > crop_len + 100]
        
        print('Paragraphs: ', len(par_list))
        
        ### Char to number
        alphabet = list(set(text))
        alphabet.sort()
        print('Found letters:', alphabet)
        char_to_number = {char: number for number, char in enumerate(alphabet)}
        number_to_char = {number: char for number, char in enumerate(alphabet)}
        
        ### Store data
        self.par_list = par_list
        self.transform = transform
        self.char_to_number = char_to_number
        self.number_to_char = number_to_char
        self.alphabet = alphabet
        self.alph_len = len(alphabet)
        
    def __len__(self):
        return len(self.par_list)
        
    def __getitem__(self, idx):
        # Get sonnet text
        text = self.par_list[idx]
        # Encode with numbers
        encoded = encode_text(self.char_to_number, text)
        # Create sample
        sample = {'text': text, 'encoded': encoded}
        # Transform (if defined)
        if self.transform:
            sample = self.transform(sample)
        return sample


def encode_text(char_to_number, text):
    i = -1
    for c in text:
        i+=1
        try:
            a = char_to_number[c]
        except:
            s = list(text)
            s[i]='?'
            text=''.join(s)
    encoded = [char_to_number[c] for c in text]
    return encoded


def decode_text(number_to_char, encoded):
    text = [number_to_char[c] for c in encoded]
    text = reduce(lambda s1, s2: s1 + s2, text)
    return text


class RandomCrop():
    
    def __init__(self, crop_len):
        self.crop_len = crop_len
        
    def __call__(self, sample):
        text = sample['text']
        encoded = sample['encoded']
        # Randomly choose an index
        tot_chars = len(text)
        start_idx = np.random.randint(0, tot_chars - self.crop_len)
        end_idx = start_idx + self.crop_len
        return {**sample,
                'text': text[start_idx: end_idx],
                'encoded': encoded[start_idx: end_idx]}
        

def create_one_hot_matrix(encoded, alphabet_len):
    # Create one hot matrix
    encoded_onehot = np.zeros([len(encoded), alphabet_len])
    tot_chars = len(encoded)
    encoded_onehot[np.arange(tot_chars), encoded] = 1
    return encoded_onehot


class OneHotEncoder():
    
    def __init__(self, alphabet_len):
        self.alphabet_len = alphabet_len
        
    def __call__(self, sample):
        # Load encoded text with numbers
        encoded = np.array(sample['encoded'])
        # Create one hot matrix
        encoded_onehot = create_one_hot_matrix(encoded, self.alphabet_len)
        return {**sample,
                'encoded_onehot': encoded_onehot}
        
                
class ToTensor():
    
    def __call__(self, sample):
        # Convert one hot encoded text to pytorch tensor
        encoded_onehot = torch.tensor(sample['encoded_onehot']).float()
        return {'encoded_onehot': encoded_onehot}
        
    
        
if __name__ == '__main__':
    
    #%% Initialize dataset
    filepath = 'Shakespeare_RomeoAndJuliet.txt'
    crop_len = 100
    dataset = RomeoDataset(filepath, crop_len)
    
    #%% Test sampling
    sample = dataset[0]
    
    print('##############')
    print('##############')
    print('TEXT')
    print('##############')
    print(sample['text'])
    
    print('##############')
    print('##############')
    print('ENCODED')
    print('##############')
    print(sample['encoded'])

    #%% Test decode function
    encoded_text = sample['encoded']
    decoded_text = decode_text(dataset.number_to_char, encoded_text)
    
    #%% Test RandomCrop
    crop_len = 100
    rc = RandomCrop(crop_len)
    sample = rc(sample)
    
    #%% Test OneHotEncoder
    alphabet_len = len(dataset.alphabet)
    ohe = OneHotEncoder(alphabet_len)
    sample = ohe(sample)

    #%% Test ToTensor
    tt = ToTensor()
    sample = tt(sample)
    print(type(sample['encoded_onehot']))
    print(sample['encoded_onehot'].shape)

    #%% Test dataloader
    
    crop_len = 100
    alphabet_len = dataset.alph_len
    trans = transforms.Compose([RandomCrop(crop_len),
                                OneHotEncoder(alphabet_len),
                                ToTensor()
                                ])
    dataset = RomeoDataset(filepath, crop_len, transform=trans)
    
    dataloader = DataLoader(dataset, batch_size=52, shuffle=True)
    
    for batch_sample in dataloader:
        batch_onehot = batch_sample['encoded_onehot']
        print(batch_onehot.shape)
    
