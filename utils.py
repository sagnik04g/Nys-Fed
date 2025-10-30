import torch
import copy
import numpy as np
import random
import argparse
import math
from datetime import datetime

# self-defined functions
from fl.client import get_clients
from fl.models import Custom_Resnet, LogisticClass, SVM_torch, Resnet18, Resnet50, LSTM_shakespeare
from data_preprocessing import get_data_dict_femnist, get_data_dict_cinic10, get_data_dict_phishing, get_data_dict_realsim, get_data_dict_w8a, get_data_dict_shakespeare

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
    parser.add_argument('--train_path_w8a', type = str, default = './data/fed_w8a_train_niid.json', help = 'w8a train path')
    parser.add_argument('--test_path_w8a' , type = str, default = './data/fed_w8a_test_niid.json'  , help = 'w8a test path')
    parser.add_argument('--train_path_femnist', type = str, default = './data/fed_femnist_train_niid_leaf.json', help = 'femnist train json path')
    parser.add_argument('--test_path_femnist' , type = str, default = './data/fed_femnist_test_niid_leaf.json'  , help = 'femnist test json path')
    parser.add_argument('--train_path_cinic10' , type = str, default = './data/fed_cinic10_train_niid.json', help = 'cinic10 train json path')
    parser.add_argument('--test_path_cinic10'  , type = str, default = './data/fed_cinic10_test_niid.json'  , help = 'cinic10 test json path')
    parser.add_argument('--train_path_realsim' , type = str, default = './data/fed_realsim_train_niid.json', help = 'realsim train json path')
    parser.add_argument('--test_path_realsim' , type = str, default = './data/fed_realsim_test_niid.json', help = 'realsim test json path')
    parser.add_argument('--train_path_shakespeare'  , type = str, default = './data/shakespeare_train_data.json'  , help = 'shakespeare train json path')
    parser.add_argument('--test_path_shakespeare'  , type = str, default = './data/shakespeare_test_data.json'  , help = 'shakespeare test json path')
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
    #parser.add_argument('-stoch','--stoch', type = bool, default = False, help = 'for stochastic or deterministic')
    parser.add_argument('-model','--model', type = str, default = 'DNN', help = 'for model type')
    parser.add_argument('-done_alpha', '--done_alpha', type = float, default = 0.0, help = 'for DONE comparison')
    parser.add_argument('-l2_reg', '--l2_reg', type = float, default = 0.001, help = 'for Nys-fed')

    
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
        case 'fednys-femnist' |'fed-ns-femnist'| 'fed-admm-femnist' | 'fednys-admm-femnist' | 'fed-femnist' | 'fed-done-femnist':
            train_data_dict = get_data_dict_femnist(args.train_path_femnist,args.min_sample)
            test_data_dict  = get_data_dict_femnist(args.test_path_femnist,args.min_sample)
            input_dim=784
            num_class=62

        case 'fednys-cinic10' | 'fed-ns-cinic10' | 'fed-admm-cinic10' | 'fednys-admm-cinic10' | 'fed-cinic10' | 'fed-done-cinic10':
            train_data_dict = get_data_dict_cinic10(args.train_path_cinic10,args.min_sample)
            test_data_dict  = get_data_dict_cinic10(args.test_path_cinic10,args.min_sample)
            input_dim=3072
            num_class=10

        case 'fednys-w8a' | 'fed-done-w8a' | 'fed-admm-w8a' | 'fednys-admm-w8a'| 'fed-w8a' | 'fed-ns-w8a':
            train_data_dict = get_data_dict_w8a(args.train_path_w8a,args.min_sample)
            test_data_dict  = get_data_dict_w8a(args.test_path_w8a,args.min_sample)
            input_dim=300
            num_class=2

        case 'fednys-phishing'| 'fed-admm-phishing' | 'fednys-admm-phishing'| 'fed-phishing' | 'fed-ns-phishing' | 'fed-done-phishing':
            train_data_dict = get_data_dict_phishing(args.train_path_phishing,args.min_sample)
            test_data_dict  = get_data_dict_phishing(args.test_path_phishing,args.min_sample)
            input_dim=68
            num_class=2
        
        case 'fednys-realsim' | 'fed-done-realsim' | 'fed-admm-realsim' | 'fednys-admm-realsim'| 'fed-realsim' | 'fed-ns-realsim' :
            train_data_dict = get_data_dict_realsim(args.train_path_realsim,args.min_sample)
            test_data_dict  = get_data_dict_realsim(args.test_path_realsim,args.min_sample)
            input_dim=20958
            num_class=2
            
        case 'fednys-shakespeare' | 'fed-admm-shakespeare' | 'fednys-admm-shakespeare'| 'fed-shakespeare' | 'fed-ns-shakespeare' | 'fed-done-shakespeare':
            train_data_dict = get_data_dict_shakespeare(args.train_path_shakespeare, args.min_sample)
            test_data_dict  = get_data_dict_shakespeare(args.test_path_shakespeare, args.min_sample)
            input_dim=80
            num_class=80

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
    if(args.model =='lstm'):
        model = LSTM_shakespeare(args)
    # if(args.model =='gru'):
    #     model = GRU_shakespeare(args)

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

        case 'fednys-cinic10' | 'fed-ns-cinic10' | 'fed-done-cinic10' | 'fed-admm-cinic10' | 'fednys-admm-cinic10' | 'fed-cinic10':
            args.min_sample = 8
            args.global_bs  = 8
            args.client_bs  = 8
            args.client_lr  = 1e-4
            args.global_lr  = 1e-4
            args.logits_lr  = 1e-4

        case 'fednys-femnist' | 'fed-ns-femnist' | 'fed-done-femnist' |'fed-admm-femnist' | 'fednys-admm-femnist' | 'fed-femnist':
            args.min_sample = 8
            args.global_bs  = 8
            args.client_bs  = 8
            args.client_lr  = 5e-3
            args.global_lr  = 5e-3
            args.logits_lr  = 5e-3

        case 'fednys-phishing' | 'fed-done-phishing' | 'fed-admm-phishing' | 'fednys-admm-phishing' | 'fed-phishing' | 'fed-ns-phishing':
            args.min_sample = 8
            args.global_bs  = 8
            args.client_bs  = 8
            args.client_lr  = 0.01
            args.global_lr  = 0.01
            args.logits_lr  = 0.01

        case 'fednys-w8a' | 'fed-admm-w8a' | 'fed-done-w8a' | 'fednys-admm-w8a' | 'fed-w8a' | 'fed-ns-w8a':
            args.min_sample = 8
            args.global_bs  = 8
            args.client_bs  = 8
            args.client_lr  = 0.01
            args.global_lr  = 0.01
            args.logits_lr  = 0.01
        
        case 'fednys-realsim' | 'fed-admm-realsim' | 'fednys-admm-realsim' | 'fed-realsim' | 'fed-ns-realsim' |'fed-done-realsim':
            args.min_sample = 64
            args.global_bs  = 64
            args.client_bs  = 64
            args.client_lr  = 1
            args.global_lr  = 1e-2
            args.logits_lr  = 1e-1
            
        case 'fednys-shakespeare' | 'fed-admm-shakespeare' | 'fednys-admm-shakespeare' | 'fed-shakespeare' | 'fed-ns-shakespeare' | 'fed-done-shakespeare':
            args.min_sample = 256
            args.global_bs  = 256
            args.client_bs  = 256
            args.client_lr  = 1
            args.global_lr  = 1e-1
            args.logits_lr  = 1e-1

        case _:
            raise Exception("wrong project:", args.project)
        
def switch_FL(args: argparse.Namespace) -> None:
    """
    Set aggregation strategy according to the choice of federated learning algorithm.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    match args.switch_FL:

        case 'FedAvg':
            args.fed_agg = 'FedAvg'

        case 'FedAdam':
            args.fed_agg = 'FedAdam'
    
        case 'FedProx':
            args.fed_agg = 'FedAvg'
            args.FedProx = True

        case 'MOON':
            args.fed_agg = 'FedAvg'
            args.MOON = True

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

def stochastic_quantization(quantized, current, prev, bitsToSend):
  """
  Performs stochastic quantization on the difference between current and previous values.

  Args:
    quantized: The previously quantized value (scalar or array).
    current: The current value (scalar or array).
    prev: The previous value (scalar or array).
    bitsToSend: The number of bits to use for quantization.

  Returns:
    The stochastically quantized value (scalar or array).
  """
  b = bitsToSend
  tau = 1 / (2**b - 1)
  # number_of_bits_toSend = 32 + len(current) * b  # The number of bits to send the value of R.

  diff = np.array(current) - np.array(prev)
  R = np.max(np.abs(diff))

  # Stochastic Quantization
  Q = (diff + R) / (2 * tau * R)
  p = Q - np.floor(Q)

  if isinstance(current, (int, float)):
    temp = np.random.rand()
    if temp <= p:
      Q_quantized = math.ceil(Q)
    else:
      Q_quantized = math.floor(Q)
  else:
    Q_quantized = np.zeros_like(Q)
    for i in range(len(current)):
      temp = np.random.rand()
      if temp <= p[i]:
        Q_quantized[i] = math.ceil(Q[i])
      else:
        Q_quantized[i] = math.floor(Q[i])

  quantized = quantized + 2 * tau * Q_quantized * R - R
  return quantized
