from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import json
from make_niid_dataset import assign_data_to_clients_niid, make_json, NumpyEncoder

data_phising_X,data_phising_Y=load_svmlight_file(r'./data/phishing')
data_phising_X=data_phising_X.toarray()

# Train-test split
X_train,X_test,Y_train,Y_test=train_test_split(data_phising_X,data_phising_Y,test_size=0.2)

# Divide the dataset among federated clients
phishing_client_datas_train=assign_data_to_clients_niid(X_train,Y_train,40,0.5)
phishing_client_datas_test=assign_data_to_clients_niid(X_test,Y_test,10,0.5)

# Convert the federated data into json format
client_train_dict_phishing = make_json(phishing_client_datas_train)
client_test_dict_phishing = make_json(phishing_client_datas_test)

with open('fed_phishing_train_niid.json','w')as f1:
    json.dump(client_train_dict_phishing,f1, cls=NumpyEncoder)
with open('fed_phishing_test_niid.json','w')as f2:
    json.dump(client_test_dict_phishing,f2, cls=NumpyEncoder)