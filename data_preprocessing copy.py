import os
import torch
import json
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def get_data_dict_mnist(json_path: str, min_sample: int = 64, image_size: int = 28) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read MNIST data pickle file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """
    # t=transforms.Compose(
    #     [transforms.Pad(18),
    #      transforms.Resize((64, 64)), transforms.ToTensor()])

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    final_data_dict={}

    if json_path.endswith(".json"):
     with open(json_path, 'r') as f:
        tmp_data_dict = json.load(f)
        for user, num_sample in zip(tmp_data_dict['users'], tmp_data_dict['num_samples']):
        # discard a user if it has too few samples
         if num_sample < min_sample:
            continue

         xs = []
         for x in tmp_data_dict['user_data'][user]['x']:
            x = torch.as_tensor(x).reshape(1, image_size, image_size)
            xs.append(x)
         xs = torch.stack(xs)
         ys = torch.as_tensor(tmp_data_dict['user_data'][user]['y']).long()
        
         final_data_dict[user] = {'x' : xs, 'y' : ys}
         
    else:
     with open(json_path, 'rb') as f:
        tmp_data_dict = pickle.load(f)
    
     for user,data in tmp_data_dict.items():
        if len(data['y']) < min_sample:
         continue

        ys_final = data['y']
        xs_final=torch.as_tensor(data['x']).float()
        ys_final=torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict

def get_data_dict_phishing(json_path: str, min_sample: int = 8) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read MNIST data pickle file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """
    # t=transforms.Compose(
    #     [transforms.Pad(18),
    #      transforms.Resize((64, 64)), transforms.ToTensor()])

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    final_data_dict={}
    
    with open(json_path, 'rb') as f:
        tmp_data_dict = json.load(f)
    
    for user,data in tmp_data_dict.items():
        if len(data['Y']) < min_sample:
         continue

        ys_final = data['Y']
        # for x in data['x']:
        #   x=np.array(x).reshape(image_size,image_size)
        #   x_img=Image.fromarray(np.uint8(x_img))
        #   x=t(x_img)
        #   xs.append(x)
        # xs_final=torch.stack(xs).float()
        xs_final=torch.as_tensor(data['X']).float()
        ys_final=torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict

def get_data_dict_w8a(json_path: str, min_sample: int = 8) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read MNIST data pickle file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """
    # t=transforms.Compose(
    #     [transforms.Pad(18),
    #      transforms.Resize((64, 64)), transforms.ToTensor()])

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    final_data_dict={}
    
    with open(json_path, 'rb') as f:
        tmp_data_dict = json.load(f)
    
    for user,data in tmp_data_dict.items():
        if len(data['Y']) < min_sample:
         continue

        ys_final = data['Y']
        xs_final=torch.tensor(data['X']).float()
        ys_final=torch.tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict

def get_data_dict_femnist(json_path: str, min_sample: int = 64, image_size: int = 28) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read MNIST data pickle file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)

    final_data_dict={}

    
    with open(json_path, 'rb') as f:
     tmp_data_dict = json.load(f)

    
    for user, num_sample in zip(tmp_data_dict['users'], tmp_data_dict['num_samples']):
    # discard a user if it has too few samples
        if num_sample < min_sample:
         continue

        xs = []
        xs = torch.as_tensor(tmp_data_dict['user_data'][user]['x'])
        ys = torch.as_tensor(tmp_data_dict['user_data'][user]['y']).long()
    
        final_data_dict[user] = {'x' : xs, 'y' : ys}

    return final_data_dict

def get_data_dict_cinic10(json_path: str, min_sample: int = 64, image_size: int = 32) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read CINIC10 data json file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    with open(json_path, 'rb') as f:
        tmp_data_dict = json.load(f)
    final_data_dict={}

    for user,data in tmp_data_dict.items():
        if len(data['Y']) < min_sample:
          continue

        ys_final = data['Y']
        xs_final = torch.as_tensor(data['X']).float()
        ys_final=torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict

def get_data_dict_realsim(json_path: str, min_sample: int = 64) -> dict[str, dict[str, torch.Tensor]]:
    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    final_data_dict={}
    with open(json_path, 'rb') as f:
        tmp_data_dict = json.load(f)
    samples=len(tmp_data_dict)
    for user,data in tmp_data_dict.items():
        if len(data['Y']) < min_sample:
          continue

        ys_final = data['Y']
        xs_final = torch.as_tensor(data['X']).float()
        ys_final = torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict
    
def get_data_dict_shakespeare(json_path: str, min_sample: int = 64, seq_length: int = 80, num_class: int = 80) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read Shakespeare data json file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        seq_length (int): length of character sequence.
        num_class (int): number of classes (unique characters) in the dataset.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # return value
    data_dict = {}

    # all 80 chars
    all_chars_sorted = ''' !"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz{}'''
    # all 75 chars (not 80 chars, because some chars are missing in train/test/both datasets)
    # all_chars_sorted   = ''' !"&'(),-.12345678:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz'''
    assert(len(all_chars_sorted) == num_class)
    total_samples=len(data['users'])
    if('train' in json_path):
       data['users'], data['num_samples']=data['users'][:int(0.9*total_samples)], data['num_samples'][:int(0.9*total_samples)]
    else:
       data['users'], data['num_samples']=data['users'][int(0.9*total_samples):], data['num_samples'][int(0.9*total_samples):]
    for user, num_sample in zip(data['users'], data['num_samples']):
        # discard a user if it has too few samples
        if num_sample < min_sample:
            continue

        xs, ys = [], []
        for x, y in zip(data['user_data'][user]['x'], data['user_data'][user]['y']):
            assert(len(x) == seq_length)
            
            y = all_chars_sorted.find(y)
            if y == -1: # cannot find character
                raise Exception('wrong character:', y)
            ys.append(y)

            seq = torch.as_tensor([all_chars_sorted.find(c) for c in x])
            xs.append(seq)

        xs = torch.stack(xs)
        ys = torch.as_tensor(ys).long()
        
        data_dict[user] = {'x' : xs, 'y' : ys}
     
    return data_dict

class Dataset(torch.utils.data.Dataset):
    """
    Self-defined dataset class.
    """

    def __init__(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        """
        Arguments:
            xs (torch.Tensor): samples.
            ys (torch.Tensor): ground truth labels.
        """

        self.xs = xs
        self.ys = ys
        
    def __len__(self) -> int:
        """
        Returns:
            (int): size of dataset.
        """

        return len(self.ys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            idx (int): index to sample.

        Returns:
            x (torch.Tensor): sample.
            y (torch.Tensor): ground truth label.
        """

        x = self.xs[idx]
        y = self.ys[idx]

        return x, y
