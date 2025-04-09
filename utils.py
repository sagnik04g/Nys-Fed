import torch
import copy
import numpy as np
import random
import argparse
from datetime import datetime

# self-defined functions
from fl.client import get_clients
from fl.models import Custom_Resnet, LogisticClass, SVM_torch, Resnet18, Resnet50
from data_preprocessing import get_data_dict_femnist, get_data_dict_cifar10, get_data_dict_phishing, get_data_dict_w8a

def seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Arguments:
        seed (int): random seed.
    """

    print('\nrandom seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
def Args() -> argparse.Namespace:
    """
    Helper function for argument parsing.

    Returns:
        args (argparse.Namespace): parsed argument object.
    """

    parser = argparse.ArgumentParser()
    
    # path parameters
    parser.add_argument('--train_path_phishing', type = str, default = './data/fed_phishing_train_niid.json', help = 'phishing train path')
    parser.add_argument('--test_path_phishing' , type = str, default = './data/fed_phishing_test_niid.json'  , help = 'phishing test path')
    parser.add_argument('--train_path_w8a', type = str, default = './data/fed_w8a_train_niid.json', help = 'phishing train path')
    parser.add_argument('--test_path_w8a' , type = str, default = './data/fed_w8a_test_niid.json'  , help = 'phishing test path')
    parser.add_argument('--train_path_femnist', type = str, default = './data/fed_femnist_train_niid_leaf.json', help = 'femnist train json path')
    parser.add_argument('--test_path_femnist' , type = str, default = './data/fed_femnist_test_niid_leaf.json'  , help = 'femnist test json path')
    parser.add_argument('--train_path_cifar10' , type = str, default = './data/fed_cifar10_train_niid.json', help = 'cifar10 train json path')
    parser.add_argument('--test_path_cifar10'  , type = str, default = './data/fed_cifar10_test_niid.json'  , help = 'cifar10 test json path')
    parser.add_argument('--input_dim'  , type = int, default = 784  , help = 'feature_dim')
    parser.add_argument('--num_class'  , type = int, default = 10  , help = 'Number of classification classes')


    # whether to use default settings for batch size and learning rates
    parser.add_argument('-d', '--default', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use default hyperparmeter settings (batch size and learning rates)')

    # general parameters for both non-FL and FL
    parser.add_argument('-p', '--project', type = str, default = 'femnist', help = 'project name, from femnist, celeba, shakespeare')
    parser.add_argument('--name', type = str, default = 'name', help = 'wandb run name')
    parser.add_argument('-seed', '--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--min_sample', type = int, default = 32, help = 'minimal amount of samples per client')
    parser.add_argument('-g_bs', '--global_bs', type = int, default = 8, help = 'batch size for global data loader')
    parser.add_argument('-c_lr', '--client_lr', type = float, default = 1e-1, help = 'client learning rate')
    parser.add_argument('--global_epoch', type = int, default = 201, help = 'number of global aggregation rounds')
    parser.add_argument('--reuse_optim', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to reuse client optimizer, should be T for non-fl and F for FL')
    parser.add_argument('-c_op', '--client_optim', default = torch.optim.SGD, help = 'client optimizer')
                    
    # general parameters for FL
    parser.add_argument('-fl', '--switch_FL', type = str, default = 'FedAgg', help = 'FL algorithm, from FedAvg, FedAdam, FedAMS, FedProx, MOON, FedAwS')
    parser.add_argument('-c_bs', '--client_bs', type = int, default = 8, help = 'batch size for client data loader')
    parser.add_argument('-C', '--client_C', type = int, default = 8, help = 'number of participating clients in each aggregation round')
    parser.add_argument('-E', '--client_epoch', type = int, default = 1, help = 'number of client local training epochs')
    
    # for FedOpt and FedAMS
    parser.add_argument('-g_lr', '--global_lr', type = float, default = 1e-3, help = 'global learning rate')
    parser.add_argument('-g_op', '--global_optim', default = torch.optim.Adam, help = 'global optimizer')
    

    parser.add_argument('--base_agg', type = str, default = 'FedAvg', help = 'basic aggregation method for non-logit layers for our method')
    parser.add_argument('--spreadout', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether conduing spread-out regularization for our method')
    parser.add_argument('--class_C', type = float, default = 1.0, help = 'proportion of classes being aggregated for our method')
    parser.add_argument('-l_lr', '--logits_lr', type = float, default = 1e-2, help = 'global learning rate for logit layer for our method')
    parser.add_argument('-l_op', '--logits_optim', default = torch.optim.Adam, help = 'global optimizer for logit layer for our method')
    parser.add_argument('-col_opt','--col_opt', type = int, default = 0, help = 'no.of columns variate optimizer for logit layer for our method')
    parser.add_argument('-lambda','--lambda', type = int, default = 0, help = 'no.of columns variate optimizer for logit layer for our method')
    parser.add_argument('-rho','--rho', type = float, default = 0, help = 'for admm method')
    parser.add_argument('-alpha','--alpha', type = float, default = 0, help = 'for admm method')
    parser.add_argument('-sketch_m','--sketch_m', type = int, default = 0, help = 'for NS method')
    parser.add_argument('-sketch_mu','--sketch_mu', type = float, default = 0, help = 'for NS method')
    parser.add_argument('-stoch','--stoch', type = bool, default = False, help = 'for stochastic or deterministic')
    parser.add_argument('-model','--model', type = str, default = 'DNN', help = 'for stochastic or deterministic')

    
    args = parser.parse_args()
    args.time = str(datetime.now())[5:-10]
    args.fed_agg = None
    args.MOON = False
    args.FedProx = False
    args.amsgrad = False
    
    return args

def get_clients_and_model(args: argparse.Namespace) -> tuple[list[object], list[object], torch.nn.Module]:
    """
    Determine dataset and model based on project name.

    Arguments:
        args (argparse.Namespace): parsed argument object.

    Returns:
        train_clients (list[Client]): list of training clients.
        test_clients (list[Client]): list of test/validation clients.
        model (torch.nn.Module): pytorch model for the specific task.
    """

    match args.project:
        case 'fednys-femnist' | 'fed-admm-femnist' | 'fednys-admm-femnist' | 'fed-femnist':
            train_data_dict = get_data_dict_femnist(args.train_path_femnist,args.min_sample)
            test_data_dict  = get_data_dict_femnist(args.test_path_femnist,args.min_sample)
            input_dim=784
            num_class=62

        case 'fednys-cifar10' | 'fed-admm-cifar10' | 'fednys-admm-cifar10' | 'fed-cifar10':
            train_data_dict = get_data_dict_cifar10(args.train_path_cifar10,args.min_sample)
            test_data_dict  = get_data_dict_cifar10(args.test_path_cifar10,args.min_sample)
            input_dim=3072
            num_class=10

        case 'fednys-w8a' | 'fed-admm-w8a' | 'fednys-admm-w8a'| 'fed-w8a' | 'fed-ns-w8a':
            train_data_dict = get_data_dict_w8a(args.train_path_w8a,args.min_sample)
            test_data_dict  = get_data_dict_w8a(args.test_path_w8a,args.min_sample)
            input_dim=300
            num_class=2

        case 'fednys-phishing' | 'fed-admm-phishing' | 'fednys-admm-phishing'| 'fed-phishing' | 'fed-ns-phishing':
            train_data_dict = get_data_dict_phishing(args.train_path_phishing,args.min_sample)
            test_data_dict  = get_data_dict_phishing(args.test_path_phishing,args.min_sample)
            input_dim=68
            num_class=2

        case _:
            raise Exception("wrong project:", args.project)
    
    if(args.model=='DNN'):
        model = Custom_Resnet(args, input_dim, num_class)
    if(args.model=='Res18'):
        model = Resnet18(args, input_dim, num_class)
    if(args.model=='Res50'):
        model = Resnet50(args, input_dim, num_class)
    if(args.model=='logistic'):
        model = LogisticClass(args, input_dim, num_class)
    if(args.model=='SVM'):
        model = SVM_torch(args, input_dim, num_class)

    # get client lists
    train_clients = get_clients(args, train_data_dict) ; del train_data_dict
    test_clients  = get_clients(args, test_data_dict) ; del test_data_dict

    # some print
    print()
    print("number of train clients:", len(train_clients))
    print("number of test  clients:", len(test_clients))
    print("length of train dataset:", sum([c.num_sample for c in train_clients]))
    print("length of test  dataset:", sum([c.num_sample for c in test_clients ]))

    return train_clients, test_clients, model

def default_setting(args: argparse.Namespace) -> None:
    """
    Set batch sizes and learning rates according to the choice of dataset and federated learning algorithm.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    assert(args.default)

    match args.project:

        case 'fednys-cifar10' | 'fed-admm-cifar10' | 'fednys-admm-cifar10' | 'fed-cifar10':
            args.min_sample = 32
            args.global_bs  = 32
            args.client_bs  = 32
            args.client_lr  = 1
            args.global_lr  = 1
            args.logits_lr  = 1

        case 'fednys-femnist' | 'fed-admm-femnist' | 'fednys-admm-femnist' | 'fed-femnist':
            args.min_sample = 16
            args.global_bs  = 16
            args.client_bs  = 16
            args.client_lr  = 1
            args.global_lr  = 1
            args.logits_lr  = 1

        case 'fednys-phishing' | 'fed-admm-phishing' | 'fednys-admm-phishing' | 'fed-phishing' | 'fed-ns-phishing':
            args.min_sample = 8
            args.global_bs  = 8
            args.client_bs  = 8
            args.client_lr  = 1
            args.global_lr  = 1
            args.logits_lr  = 1

        case 'fednys-w8a' | 'fed-admm-w8a' | 'fednys-admm-w8a' | 'fed-w8a' | 'fed-ns-w8a':
            args.min_sample = 16
            args.global_bs  = 16
            args.client_bs  = 16
            args.client_lr  = 1
            args.global_lr  = 1
            args.logits_lr  = 1

        case _:
            raise Exception("wrong project:", args.project)
        
def switch_FL(args: argparse.Namespace) -> None:
    """
    Set aggregation strategy according to the choice of federated learning algorithm.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    match args.switch_FL:

        case 'FedAgg':
            args.fed_agg = 'FedAgg'

        case 'FedAdam':
            args.fed_agg = 'FedOpt'

        case 'FedAMS':
            args.fed_agg = 'FedOpt'
            args.amsgrad = True
    
        case 'FedProx':
            args.fed_agg = 'FedAgg'
            args.FedProx = True

        case 'MOON':
            args.fed_agg = 'FedAgg'
            args.MOON = True

        case 'FedAwS':
            args.fed_agg = 'FedAwS'

        case 'FedNew':
            args.fed_agg = 'FedNew'

        case 'FedNS':
            args.fed_agg = 'FedNS'
            
        case _:
            raise Exception("wrong switch_FL:", args.switch_FL)
    
def weighted_avg_params(params: list[dict[str, torch.Tensor]], weights: list[int] = None) -> dict[str, torch.Tensor]:
    """
    Compute weighted average of client models.

    Argument:
        params (list[dict[str, torch.Tensor]]): client model parameters. Each element in this list is the state_dict of a client model.
        weights (list[int]): weight per client. Each element in this list is the number of samples of a client.

    Returns:
        params_avg (dict[str], torch.Tensor): averaged global model parameters (state_dict), which can be loaded using global_model.load_state_dict.
    """

    if weights == None:
        weights = [1.0] * len(params)
        
    params_avg = copy.deepcopy(params[0])
    for key in params_avg.keys():
        params_avg[key] *= weights[0]
        for i in range(1, len(params)):
            params_avg[key] += params[i][key] * weights[i]
        params_avg[key] = torch.div(params_avg[key], sum(weights))
    return params_avg

def weighted_avg(values: any, weights: any) -> any:
    """
    Calculate weighted average of a vector of values.

    Arguments:
        values (any): values. Can be list, torch.Tensor, numpy.ndarray, etc.
        weights (any): weights. Can be list, torch.Tensor, numpy.ndarray, etc.

    Returns:
        any: weighted average value.
    """

    sum_values = 0
    for v, w in zip(values, weights):
        sum_values += v * w
    return sum_values / sum(weights)