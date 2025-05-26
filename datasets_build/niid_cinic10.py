from make_niid_dataset import assign_data_to_clients_niid, make_data_from_images,make_json, NumpyEncoder
import json

train_images_path = r"/home/ub/Downloads/CINIC10_dataset/train"
test_images_path = r"/home/ub/Downloads/CINIC10_dataset/test"
cinic10_train_X,cinic10_train_Y = make_data_from_images(train_images_path)
cinic10_test_X,cinic10_test_Y = make_data_from_images(test_images_path)

# Divide the dataset among federated clients
cinic10_client_datas_train=assign_data_to_clients_niid(cinic10_train_X,cinic10_train_Y,128,0.5)
cinic10_client_datas_test=assign_data_to_clients_niid(cinic10_test_X,cinic10_test_Y,40,0.5)

# Convert the federated data into json format
client_train_dict_cinic10 = make_json(cinic10_client_datas_train)
client_test_dict_cinic10 = make_json(cinic10_client_datas_test)

with open('fed_cinic10_train_niid.json','w') as f1:
    json.dump(client_train_dict_cinic10,f1, cls=NumpyEncoder)
with open('fed_cinic10_test_niid.json','w') as f2:
    json.dump(client_test_dict_cinic10,f2, cls=NumpyEncoder)