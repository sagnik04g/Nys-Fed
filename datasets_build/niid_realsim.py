from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import json
from make_niid_dataset import assign_data_to_clients_niid, make_json, NumpyEncoder

data_realsim_X,data_realsim_Y=load_svmlight_file(r'./data/real-sim')
data_realsim_X=data_realsim_X.toarray()

# Train-test split
X_train,X_test,Y_train,Y_test=train_test_split(data_realsim_X,data_realsim_Y,test_size=0.2)

# Divide the dataset among federated clients
realsim_client_datas_train=assign_data_to_clients_niid(X_train,Y_train,40,0.5)
realsim_client_datas_test=assign_data_to_clients_niid(X_test,Y_test,10,0.5)

# Convert the federated data into json format
client_train_dict_realsim = make_json(realsim_client_datas_train)
client_test_dict_realsim = make_json(realsim_client_datas_test)

with open('fed_realsim_train_niid.json','w')as f1:
    json.dump(client_train_dict_realsim,f1, cls=NumpyEncoder)
with open('fed_realsim_test_niid.json','w')as f2:
    json.dump(client_test_dict_realsim,f2, cls=NumpyEncoder)