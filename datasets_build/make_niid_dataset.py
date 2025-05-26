import numpy as np
import random
import json
import os
from PIL import Image

def make_data_from_images(images_path):
    X, Y=[], []
    classes=os.listdir(images_path)
    class_dict={class_name:i for i,class_name in enumerate(classes)}
    for class_name in classes:
        if(os.path.isdir(os.path.join(images_path,class_name))):
            images_name=os.listdir(os.path.join(images_path,class_name))
            for img in images_name:
                image=np.array(Image.open(os.path.join(images_path,class_name,img)))
                if(len(image.shape)==2):
                    image=np.repeat(image,3)
                image=image.reshape(32,32,3)
                X.append(image)
                Y.append(class_dict[class_name])
    return X,Y


def make_json(client_dataset):
 client_data_dict={}
 user=0
 for X,y in client_dataset:
    if(X.shape[0]!=0):
        client_data_dict['user'+str(user)]={'X':X,'Y':y}
        user+=1
 return client_data_dict

def assign_data_to_clients_niid(train_dataset_X,train_dataset_Y, no_of_clients,alpha):
    client_indices = partition_data(train_dataset_X, train_dataset_Y, no_of_clients, alpha)
    client_datasets = [(train_dataset_X[indices],train_dataset_Y[indices]) for indices in client_indices]
    return client_datasets

def partition_data(dataset_X,dataset_Y, num_clients, alpha):
    data_indices = np.arange(len(dataset_X))
    targets = np.array(dataset_Y)
    num_classes = len(np.unique(targets))

    # Create Dirichlet distribution
    class_distribution = np.random.dirichlet(alpha=[alpha] * num_clients, size=num_classes)

    client_data_indices = [[] for _ in range(num_clients)]
    for class_idx, class_dist in enumerate(class_distribution):
        class_indices = data_indices[targets == class_idx]
        np.random.shuffle(class_indices)
        split_indices = np.array_split(class_indices, [int(np.round(val)) for val in np.cumsum(class_dist[:-1]) * len(class_indices)])
        for client_idx, client_indices in enumerate(split_indices):
            client_data_indices[client_idx].extend(client_indices)

    return client_data_indices

def assign_data_to_clients_niid(train_dataset_X,train_dataset_Y, no_of_clients,alpha):
    client_indices = partition_data(train_dataset_X, train_dataset_Y, no_of_clients, alpha)
    client_datasets = [(train_dataset_X[indices],train_dataset_Y[indices]) for indices in client_indices]
    return client_datasets


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return super().default(obj)
    

# phishing_client_datas=assign_data_to_clients_niid(data_phising_X,data_phising_Y,100,0.5)
# phishing_client_data_train=phishing_client_datas[:70]
# phishing_client_data_test=phishing_client_datas[70:]
# client_train_dict_phishing = make_json(phishing_client_data_train)
# client_test_dict_phishing = make_json(phishing_client_data_test)

# import json
# with open('fed_phishing_train_niid.json','w')as f1:
#     json.dump(client_train_dict_phishing,f1, cls=NumpyEncoder)
# with open('fed_phishing_test_niid.json','w')as f2:
#     json.dump(client_test_dict_phishing,f2, cls=NumpyEncoder)