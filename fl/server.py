import numpy  as np
import torch
import math
import tqdm
import copy
import wandb
from fl.client import Client
from fl.models import model_eval, cal_metrics
from utils import weighted_avg_params, weighted_avg
from torchmetrics.functional import pairwise_cosine_similarity

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# FedAwS cosine similarity margin
margin = 0
    
def federated_learning(args: object, train_clients: list[object], test_clients: list[object], global_model: torch.nn.Module) -> None:
    """
    Main loop for federated learning.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        train_clients (list[Client]): training clients.
        test_clients (list[Client]): test / validation clients.
        global_model (torch.nn.Module): pytorch model (global model on the server).
    """

    # determine how many clients are updated per global round
    num_train_client  = len(train_clients)
    if args.client_C < 1.0: # proportion
        num_update_client = min(max(math.ceil(args.client_C * num_train_client), 1), num_train_client) # number of clients to update per round
    else: # client_C itself is num_update_client
        num_update_client = min(args.client_C, num_train_client)
    print("\n Number of updated clients per round during training:", num_update_client)
    
    # global optimizer
    global_model.to(device)
    global_optim = args.global_optim(global_model.parameters(), lr = args.global_lr, amsgrad = args.amsgrad)
    logits_optim = args.logits_optim(global_model.logits.parameters(), lr = args.logits_lr, eps = 1e-5)

    # train-valid-test split on server level
    global_train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in train_clients])
    global_test_dataset  = torch.utils.data.ConcatDataset([c.dataset for c in test_clients ])
    global_train_loader  = torch.utils.data.DataLoader(global_train_dataset, batch_size = args.global_bs, shuffle = False)
    global_test_loader   = torch.utils.data.DataLoader(global_test_dataset , batch_size = args.global_bs, shuffle = False)
    
    # performance before training
    wandb_log = {}
    model_eval(global_model, global_train_loader, wandb_log, 'train/', args.model)
    model_eval(global_model, global_test_loader , wandb_log, 'test/', args.model )
    wandb.log(wandb_log)
    
    # for MOON
    previous_features = None
    
    # global round loop
    print()

    ## ADMM lambda_i for clients
    
    model_param_group = list(global_optim.param_groups)[0]['params']
    
    client_lambda = [torch.zeros(torch.cat([pi.view(-1) for pi in model_param_group]).shape[0],1).to(device) for _ in range(num_update_client)]
    y_k =torch.zeros(torch.cat([pi.view(-1) for pi in model_param_group]).shape[0],1).to(device)
    
    yi_k = [torch.zeros(torch.cat([pi.view(-1) for pi in model_param_group]).shape[0],1).to(device) for _ in range(num_update_client)]
    init_epoch=0
    


    for current_global_epoch in tqdm.tqdm(range(args.global_epoch)):
        # select clients which are updated in this round
        update_clients = np.random.choice(train_clients, num_update_client, replace = False)
        client_weights = [c.num_sample for c in update_clients]
        client_models  = [copy.deepcopy(global_model) for c in update_clients]
        client_sketches = []
        sketch_mu = args.sketch_mu
        client_iter = 0    

        # training
        for client, client_model in zip(update_clients, client_models):

                   
            if(args.col_opt != 0 or args.alpha!=0 or args.rho!=0):
                client.local_train_second_order(client_model, y_k, client_lambda[client_iter], args)
                

            elif(args.sketch_m != 0):
                sketch_matrix = client.local_train_second_order(client_model, y_k, client_lambda[client_iter], args)
                client_sketches.append(sketch_matrix)

            else:
                client.local_train_LBFGS(client_model, global_model, args)
            
            client_iter+=1

            
        init_epoch+=1
        yi_k=[torch.cat([p.grad.view(-1,1) for p in m.parameters()]).to(device) for m in client_models]
        
    
        y_k=sum(yi_k)/len(yi_k)
        # print("Actual y_k:" + str(torch.Tensor.norm(y_k)) + " Expected y_k:" + str(torch.Tensor.norm(y_k_check)))
        # print("Average of client weights norm: " + str(torch.Tensor.norm(sum([torch.cat([p.view(-1) for p in m.parameters()]) for m in client_models])/len(client_models))))
        # print("\n Global model weight norm: "+ str(torch.Tensor.norm(torch.cat([p.view(-1) for p in global_model.parameters()]))))
        
        # PS update
        eval(args.fed_agg)(global_model, client_models, client_weights, client_sketches, sketch_mu, # basic FL parameters
                           global_optim, y_k, # for FedOpt (FedAdam and FedAMS)
                           logits_optim, # for FedAwS 
                           current_global_epoch, args.global_epoch, args.class_C, args.base_agg, args.spreadout)

        # lambda_i update for one-pass admm step
        for client_id in range(len(client_lambda)):
            client_lambda[client_id] += (args.rho * (yi_k[client_id] - y_k)).to(device)
        
        torch.cuda.empty_cache()
        # stability
        for p in global_model.parameters():
            torch.nan_to_num_(p.data, nan=1e-9, posinf=1e-9, neginf=1e-9)

        # performance metrics
        global_train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in update_clients])
        global_train_loader  = torch.utils.data.DataLoader(global_train_dataset, batch_size = args.global_bs, shuffle = False)
        wandb_log = {}
        model_eval(global_model, global_train_loader, wandb_log, 'train/', args.model)
        model_eval(global_model, global_test_loader , wandb_log, 'test/', args.model)
        wandb.log(wandb_log)
        
    # global_model.to('cpu')
    # wandb.finish()

def server_eval(clients: list[object], wandb_log: dict[str, float], metric_prefix: str) -> None:
    """
    (Obsolete.) Evaluate model performance globally by letting each client conduct inference locally and then collecting all inferences and calculating metrics.

    Arguments:
        clients (list[Client]): list of clients.
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
    """

    labels = []
    preds  = []
    for c in clients:
        l, p = c.local_eval()
        labels.append(l)
        preds .append(p)
    labels = torch.cat(labels)
    preds  = torch.cat(preds )
    cal_metrics(labels, preds, wandb_log, metric_prefix)    

# FedNew for ADMM method
def FedNew(global_model: torch.nn.Module, client_models: list[torch.nn.Module], client_weights: list[int], client_sketches: list[torch.Tensor], sketch_mu: float, global_optim: torch.optim, y_k, *_) -> None:

    
    # global_model.train()
    # ls = 0
    # for p_name, p in global_model.named_parameters():
    #     if p.requires_grad:
    #         p.grad = global_model.state_dict()[p_name] -  y_k[ls:ls+p.numel()].view(p.shape)  # x_k+1 = x_k - y_k
    #     ls += torch.numel(p)
    # global_optim.step()
    # global_optim.zero_grad()
    client_params = [m.state_dict() for m in client_models]
    client_weights = [1 for _ in client_weights]
    new_global_params = weighted_avg_params(params = client_params, weights = client_weights)
    global_model.load_state_dict(new_global_params)

def FedNS(global_model: torch.nn.Module, client_models: list[torch.nn.Module], client_weights: list[int], client_sketches: list[torch.Tensor], sketch_mu: float, global_optim: torch.optim, *_) -> None:

    # new_global_state_dict = copy.deepcopy(global_model.state_dict())
    global_hessian=torch.zeros(client_sketches[0].shape[1],client_sketches[0].shape[1]).to(device)
    global_grads=torch.zeros_like(torch.cat([p.grad.view(-1) for p in client_models[0].parameters()])).to(device)
    
    for i in range(len(client_sketches)):
        print('client weight: '+str(client_weights[i]))
        print('client sketch matrix shape: '+str(client_sketches[i].shape))
        global_hessian+=client_weights[i]*torch.mm(client_sketches[i].T , client_sketches[i]).to(device)
        client_grad = torch.cat([p.grad.view(-1) for p in client_models[i].parameters()]).to(device)
        global_grads+=client_weights[i]*client_grad

    
    ls=0
    vk = sketch_mu*(torch.linalg.pinv(global_hessian)@global_grads).to(device)
    global_model.train()
    for p_name, p in global_model.named_parameters():
        if p.requires_grad:
            p.grad =  vk[ls:ls+torch.numel(p)].view(p.shape)
        ls += torch.numel(p)
    # global_model.load_state_dict(new_global_state_dict)
    global_optim.step()
    global_optim.zero_grad()

def FedAgg(global_model: torch.nn.Module, client_models: list[torch.nn.Module], client_weights: list[int],  *_) -> None:
    """
    Federated learning algorithm FedAvg.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
    """

    client_params = [m.state_dict() for m in client_models]
    new_global_params = weighted_avg_params(params = client_params, weights = client_weights)
    global_model.load_state_dict(new_global_params)

def FedOpt(global_model: torch.nn.Module, client_models: list[torch.nn.Module], client_weights: list[int], global_optim: torch.optim, *_) -> None:
    """
    Federated learning algorithm FedOpt. Depending on the choice of optimizer, it can be deviated into different variates like FedAdam and FedAMS.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        global_optim (torch.optim): pytorch optimizer for global model.
    """

    client_params  = [m.state_dict() for m in client_models]
    new_global_params = weighted_avg_params(params = client_params, weights = client_weights)
    
    # pseudo-gradient
    global_model.train()
    for p_name, p in global_model.named_parameters():
        if p.requires_grad:
            p.grad = global_model.state_dict()[p_name] - new_global_params[p_name].to(p.device)
    
    # apply optimizer
    global_optim.step()
    global_optim.zero_grad()

def FedAwS(global_model: torch.nn.Module, 
           client_models: list[torch.nn.Module], 
           client_weights: list[int], 
           global_optim: torch.optim, 
           logits_optim: torch.optim, 
           *_) -> None:
    """
    Federated learning algorithm FedAwS.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        global_optim (torch.optim): (useless) pytorch optimizer for global model.
        logits_optim (torch.optim): pytorch optimizer for logit layer of global model.
    """

    FedAvg(global_model, client_models, client_weights)
    global_model.train()
    
    # spreadout regularizer
    wb = torch.cat((global_model.logits.weight, global_model.logits.bias.view(-1, 1)), axis = 1)
    cos_sim_mat = pairwise_cosine_similarity(wb)
    cos_sim_mat = (cos_sim_mat > margin) * cos_sim_mat
    loss = cos_sim_mat.sum() / 2
    loss.backward()
    
    # apply optimizer
    logits_optim.step()
    logits_optim.zero_grad()
