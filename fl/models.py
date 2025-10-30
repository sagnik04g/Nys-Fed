import random
import psutil
import gpustat
from fl.fed_ns import NewtonSketch
from fl.nys_admm import NYS_ADMM
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from typing import List
import torch.nn.functional as F
import copy
import os
import time
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix
from fl.fed_admm import ADMM
from fl.fed_done import DONE
# GPU
if torch.cuda.is_available():
    cuda_id=int(os.environ.get("CUDA_VISIBLE_DEVICES"))
    device='cuda'    
else:
    device ='cpu'

# metrics that require average parameter
metrics_with_avg = {'prec' : precision_score, 'recl' : recall_score, 'f1' : f1_score}
avg = 'macro'
current_time = time.time()
memory_info=[0]
# metrics that dont require average parameter
metrics_no_avg = {'accu' : accuracy_score, 'mcc' : matthews_corrcoef}

# a list that contains resnet18 during runtime
resnet18_list=[]
resnet50_list=[]

# for FedProx
FedProx_mu = 0.01

# for MOON
MOON_temperature = 0.5
MOON_mu = 1.0

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Define the convolutional layers in the residual block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If the input and output dimensions are not the same, we need to change the input
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Apply the first convolution, batch norm, and ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Apply the second convolution and batch norm
        out = self.bn2(self.conv2(out))
        
        # Add the skip connection (shortcut) to the output
        out += self.shortcut(x)
        
        # Apply ReLU activation to the output after adding the shortcut
        out = F.relu(out)
        
        return out

class LogisticClass(nn.Module):
    def __init__(self, args: object, input_dim:int = 68, num_class: int = 2) -> None:
        super(LogisticClass, self).__init__()
        self.logits = nn.Linear(input_dim, num_class)
        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None
        self.binary = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        h = x.view(-1, x[0].flatten().shape[0])  # Flatten the input tensor 
        
        # Apply sigmoid activation function
        x = torch.sigmoid(self.logits(h))  # Output layer
        return x,h

class SVM_torch(nn.Module):
    def __init__(self, args: object, input_dim:int = 68, num_class: int = 2) -> None:
        super(SVM_torch, self).__init__()
        # self.fc1 = nn.Linear(input_dim,12)
        self.logits = nn.Linear(input_dim, 2)
        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None
        self.binary = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x.view(-1, x[0].flatten().shape[0])  # Flatten the input tensor 
        # h = self.fc1(x)
        x = self.logits(h)  # Output layer
        return x,h

class Resnet18(torch.nn.Module):
    """
    (Obsolete.) Resnet model for Covid-19 dataset.
    """

    def __init__(self, args: object, input_dim: int = 784, num_class: int = 10) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            num_class (int): number of classes in the dataset.
            freeze (bool): (obsolete) whether conducting transfer learning or finetuning.
        """

        super(Resnet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = torch.nn.Identity()
        if(args.done_alpha!=0 or 'fed-admm' in args.project or 'fed-ns' in args.project):
            for p in self.resnet18.parameters():                # freeze resnet50
                p.requires_grad = False
        self.logits = torch.nn.Linear(in_features = 512, out_features = num_class)

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """
        if(x[0].flatten().shape[0]==784):
            x=x.reshape(-1,1,28,28)
            x=x.repeat(1,3,1,1)
            x = F.pad(x,(18,18,18,18),mode='constant', value=0)
        if(x[0].flatten().shape[0]==3072):
            x=x.reshape(-1,3,32,32)
            x = F.pad(x,(16,16,16,16),mode='constant', value=0)
        h = self.resnet18(x)
        x = self.logits(h)
        return x, h

class Resnet50(torch.nn.Module):
    """
    (Obsolete.) Resnet model for Covid-19 dataset.
    """

    def __init__(self, args: object, input_dim:int = 784, num_class: int = 62) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            num_class (int): number of classes in the dataset.
            freeze (bool): (obsolete) whether conducting transfer learning or finetuning.
        """

        super(Resnet50, self).__init__()

        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.fc = torch.nn.Identity()
        if(args.done_alpha!=0 or 'fed-admm' in args.project or 'fed-ns' in args.project):
            for p in self.resnet50.parameters():                # freeze resnet50
                p.requires_grad = False
        self.logits = torch.nn.Linear(in_features = 2048, out_features = num_class)
        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None
        self.binary = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """    
        if(x[0].flatten().shape[0]==784):
            x=x.reshape(-1,1,28,28)
            x=x.repeat(1,3,1,1)
            x = F.pad(x,(18,18,18,18),mode='constant', value=0)
        if(x[0].flatten().shape[0]==3072):
            x=x.reshape(-1,3,32,32)
            x = F.pad(x,(16,16,16,16),mode='constant', value=0)
        h = self.resnet50(x)
        x = self.logits(h)
        return x, h

class Custom_Resnet(torch.nn.Module):
    """
    (Obsolete.) Resnet model for Covid-19 dataset.
    """

    def __init__(self, args: object, input_dim:int = 784, num_class: int = 62) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            image_size (int): height / width of images. The images should be of rectangle shape.
            num_class (int): number of classes in the dataset.
        """

        super(Custom_Resnet, self).__init__()
        if input_dim==3072:
         input_channels=3
         out_dim=512
        if input_dim==784:
         input_channels=1
         out_dim=288

        self.encoder = torch.nn.Sequential(
        ResidualBlock(input_channels,8, stride=1),
        nn.BatchNorm2d(8),
        nn.LeakyReLU(0.1),
        ResidualBlock(8, 16, stride=2),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.1),
        ResidualBlock(16, 32, stride=2),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Flatten(), # Downsample
        nn.LeakyReLU(0.1),
        # Final fully connected layer (for example, for classification)
        nn.Linear(out_dim, 2048),
        nn.LeakyReLU(0.1))
        if(args.done_alpha!=0 or 'fed-admm' in args.project or 'fed-ns' in args.project):
            for p in self.encoder.parameters():                # freeze resnet50
                p.requires_grad = False
        self.logits = nn.Linear(2048, num_class)
        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """
        if x[0].flatten().shape[0]==3072:
         x=x.reshape(-1,3,32,32)
        elif x[0].flatten().shape[0]==784:
         x=x.reshape(-1,1,28,28)
        elif x[0].flatten().shape[0]==300:
         zero_padding=torch.zeros(x.shape[0],100).to(device)
         x=torch.cat(((x.view(-1,300)),zero_padding),dim=1).to(device)
         x=x.reshape(-1,1,20,20)
        elif x[0].flatten().shape[0]==68:
         zero_padding=torch.zeros(x.shape[0],32).to(device)
         x=torch.cat(((x.view(-1,68)),zero_padding),dim=1).to(device)
         x=x.reshape(-1,1,10,10)
        h = self.encoder(x)
        x = self.logits(h)
        return x, h
        
class LSTM_shakespeare(torch.nn.Module):
    """
    LSTM model for Shakespeare dataset. The model structure follows the LEAF framework.
    """

    def __init__(self, args: object, embedding_dim: int = 8, hidden_size: int = 256, num_class: int = 80) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            embedding_dim (int): dimension of character embedding.
            hidden_size (int): dimension of LSTM hidden state.
            num_class (int): number of classes (unique characters) in the dataset.
        """

        super(LSTM_shakespeare, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings = num_class, embedding_dim = embedding_dim)
        self.encoder   = torch.nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = 2, batch_first = True)
        if(args.done_alpha!=0 or 'fed-admm' in args.project or 'fed-ns' in args.project):
            for p in self.encoder.parameters():                # freeze resnet50
                p.requires_grad = False
        self.logits    = torch.nn.Linear(in_features = hidden_size, out_features = num_class)

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): logits (not softmaxed yet).
            h (torch.Tensor): latent features (useful for tSNE plot and some FL algorithms).
        """

        x = self.embedding(x)
        x, (hn, cn) = self.encoder(x)
        h = x[:, -1, :]
        x = self.logits(h)
        return x, h
    
def custom_multi_margin_loss(x, y, margin=1.0):
    
        n = x.size(0)
        num_classes = x.size(1)
        loss = torch.zeros(n)
        for i in range(n):
            correct_class_score = x[i, y[i]]
            incorrect_class_losses = torch.relu(margin - correct_class_score + x[i, :])
            # incorrect_class_losses[y[i]] = 0
            loss[i] = torch.sum(incorrect_class_losses)

        return loss

def model_train_second_order(model: torch.nn.Module, y_k: torch.Tensor, data_loader: torch.utils.data.DataLoader, num_client_epoch: int,  col_opt: int, rho: float, alpha :float, lambda_i: float, sketch_m: int, model_type: str, sketch_mu: float, done_alpha: float, d_i_previous, l2_reg) -> None:
    
    model.train()
    optim = model.optim(model.parameters(), lr = model.lr, momentum=0.5)
    d_i_k=None
    hessian_sketch=None
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    params_shape=torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad]).shape[0]
    # condition for fed-NS
    if(sketch_m!=0 and col_opt==0):
        fed_ns_pre = NewtonSketch(trainable_params, sketch_m, sketch_mu, device)
        hessian_sketch = fed_ns_pre.compute_hessian_sketch(data_loader,model, model_type)
    # for stochastic update
    if col_opt!=0:
        preconditioner = NYS_ADMM(trainable_params,col_opt, lambda_i, l2_reg, y_k, alpha, rho, -1, device)
        max_eigen,min_eigen = preconditioner.compute_hessian(data_loader, model, model_type)
        print(f"Max eigen val: {max_eigen}, Min eigen val: {min_eigen}")
    if(alpha!=0 and col_opt==0):
        preconditioner = ADMM(trainable_params,lambda_i, y_k, alpha, rho, device)
        preconditioner.compute_hessian(data_loader, model, model_type)
    if(done_alpha!=0 and col_opt==0):
        preconditioner = DONE(trainable_params, d_i_previous, done_alpha, device)
        d_i_k=preconditioner.compute_hessian(data_loader, model, model_type)
    for inputs, targets in data_loader:
        ## Picking a random batch
        inputs,targets = inputs.to(device), targets.to(device)
        outputs, _ = model(inputs)
        if(model_type == 'SVM'):
            weights = model.logits.weight.view(-1,1).squeeze()
            # loss = F.multi_margin_loss(outputs, targets) + (0.1 * torch.sum(weights**2))
            # loss = torch.mean(torch.relu(1-(outputs*targets.reshape(-1,1))**2)) + (0.1 * torch.sum(weights**2))
            loss = torch.mean(custom_multi_margin_loss(outputs,targets)) + (0.1 * torch.sum(weights**2))
        else:
            loss = F.cross_entropy(outputs, targets)
            #if(done_alpha!=0):
            #    loss = -loss
        optim.zero_grad()
        loss.backward()
        ## For Second-Order gradient calculation
        if(sketch_m==0):
         preconditioner.step()
        optim.step()
        
    if(device=='cuda'):
        try:
            gpus = gpustat.GPUStatCollection.new_query()
            processes = gpus[cuda_id].processes
            for process in processes:
                if(model_type in process['full_command']):
                    memory_usage=float(process['gpu_memory_usage'])
                else:
                    continue
        except Exception as e:
            print(f"Error getting GPU info: {e}")
    else:
        memory_usage = psutil.Process().memory_info().rss/(1024*1024)
    memory_info.clear()
    memory_info.append(memory_usage)         # stability
    for p in model.parameters():
        torch.nan_to_num_(p.data, nan=1e-9, posinf=1e-9, neginf=1e-9)

    torch.cuda.empty_cache()
    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())
    if hessian_sketch is not None and sketch_m!=0:
        return hessian_sketch
    if d_i_k is not None:
        return d_i_k
    return

def model_train_first_order(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_client_epoch: int, args) -> None:
    """
    Train a model when FedProx is chosen for federated learning.

    Arguments:
        model (torch.nn.Module): pytorch model (client model).
        global_model (torch.nn.Module): pytorch model (global model).
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        num_client_epoch (int): number of training epochs.
    """

    process = psutil.Process()
    mem_info = process.memory_info()
    model.train()
    optim = model.optim(model.parameters(), lr = model.lr, momentum=0.8)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    for current_client_epoch in range(num_client_epoch):
        for batch_id, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            p, _ = model(x)
            loss = F.cross_entropy(p, y)
            optim.zero_grad()

            loss.backward()
            optim.step()
            
            # stability
            for p in model.parameters():
                torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)
    if(device=='cuda'):
        try:
            gpus = gpustat.GPUStatCollection.new_query()
            processes = gpus[cuda_id].processes
            for process in processes:
                if(process['full_command'][3]==args.project):
                    memory_usage=float(process['gpu_memory_usage'])
                else:
                    memory_usage=0.0
        except Exception as e:
            print(f"Error getting GPU info: {e}")
    else:
        memory_usage = psutil.Process().memory_info().rss/(1024*1024)
    memory_info.clear()
    memory_info.append(memory_usage)     # stability
    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())
    

def model_train_FedProx(model: torch.nn.Module, global_model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, num_client_epoch: int) -> None:
    """
    Train a model when FedProx is chosen for federated learning.

    Arguments:
        model (torch.nn.Module): pytorch model (client model).
        global_model (torch.nn.Module): pytorch model (global model).
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        num_client_epoch (int): number of training epochs.
    """

    # for resnet18
    # if isinstance(model, Resnet18_mnist):
    #     resnet18_list[0].train()

    # if isinstance(model, Resnet50_cifar10):
    #     resnet50_list[0].train()

    model.train()
    optim = model.optim(model.parameters(), lr = model.lr)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    for current_client_epoch in range(num_client_epoch):
        for batch_id, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            p, _ = model(x)
            loss = F.cross_entropy(p, y)
            
            # FedProx
            for p1, p2 in zip(model.parameters(), global_model.parameters()):
                ploss = (p1 - p2.detach()) ** 2
                loss += FedProx_mu * ploss.sum()

            loss.backward()
            optim.step()
            optim.zero_grad()

            # stability
            for p in model.parameters():
                torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())

def model_train_MOON(model: torch.nn.Module, global_model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, previous_features: torch.Tensor) -> torch.Tensor:
    """
    Train a model when MOON is chosen for federated learning.

    Arguments:
        model (torch.nn.Module): pytorch model (client model).
        global_model (torch.nn.Module): pytorch model (global model).
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        previous_features (torch.Tensor): features extracted by client model in last global epoch.

    Returns:
        total_features (torch.Tensor): features extracted by client model in current global epoch.
    """

    model.train()
    optim = model.optim(model.parameters(), lr = model.lr)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    cos = torch.nn.CosineSimilarity(dim=-1)

    for batch_id, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        
        # feed into model
        p, features = model(x)
        if batch_id == 0:
            total_features = torch.empty((0, features.size()[1]), dtype=torch.float32).to(device)
        total_features = torch.cat([total_features, features], dim=0)
        loss = F.cross_entropy(p, y)

        # for MOON
        features_tsne = np.squeeze(features)
        _, global_feat = global_model(x)
        global_feat_copy = copy.copy(global_feat)
        posi = cos(features_tsne, global_feat_copy.to(device))
        logits = posi.reshape(-1,1)
        if previous_features == None or torch.count_nonzero(previous_features) == 0:
            previous_features = torch.zeros_like(features_tsne)
            nega = cos(features_tsne, previous_features)
            logits = torch.cat((posi.reshape(-1,1), nega.reshape(-1,1)), dim=1)
        if previous_features.dim() == 3:
            for prev_feat in previous_features[:, batch_id*y.size()[0]:(batch_id+1)*y.size()[0], :]:
                prev_nega = cos(features_tsne.to(device),prev_feat.to(device))
                logits = torch.cat((logits, prev_nega.reshape(-1,1)), dim=1)
        
        logits /= MOON_temperature # 0.5
        cos_labels = torch.zeros(logits.size(0)).long().to(device)
        loss_contrastive = F.cross_entropy(logits, cos_labels)
        if torch.count_nonzero(previous_features) != 0:
            loss += MOON_mu * loss_contrastive

        loss.backward()
        optim.step()
        optim.zero_grad()
        
        # stability
        for p in model.parameters():
            torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())
    
    return total_features

def model_eval(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               wandb_log: dict[str, float], 
               metric_prefix: str = 'prefix/',
               model_type: str = '', 
               returns: bool = False,
               ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Evaludate the performance of a model with differnt metrics (loss, accuracy, MCC score, precision, recall, F1 score).

    Arguments:
        model (torch.nn.Module): pytorch model.
        data_loader (torch.utils.data.DataLoader): pytorch data loader.
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
        returns (bool): whether to return ground truth labels and logits, or to calculate metrics

    Returns:
        epoch_labels (torch.Tensor): ground truth labels.
        epoch_predicts (torch.Tensor): logits (not softmaxed yet).

    """

    model.eval()
    epoch_labels   = []
    epoch_predicts = []
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                
                p, _ = model(x)
                
                epoch_labels.append(y)
                epoch_predicts.append(p)
   
    epoch_labels   = torch.cat(epoch_labels).detach().to('cpu')
    epoch_predicts = torch.cat(epoch_predicts).detach().to('cpu')
    
    if returns:
        return epoch_labels, epoch_predicts
    else:
        if(model_type=='SVM'):
            loss = custom_multi_margin_loss(epoch_predicts, epoch_labels)
            wght=model.logits.weight.view(-1,1)
            loss+=(0.01 * torch.sum(wght**2)).to('cpu')
            
        else:
            loss = F.cross_entropy(epoch_predicts, epoch_labels)
        cal_metrics(epoch_labels, model_type, epoch_predicts, wandb_log, metric_prefix, model.binary, loss)

def cal_metrics(labels: torch.Tensor, model_type: str, preds: torch.Tensor, wandb_log: dict[str, float], metric_prefix: str, binary: bool, loss) -> None:
    """
    Compute metrics (loss, accuracy, MCC score, precision, recall, F1 score) using ground truth labels and logits.

    Arguments:
        labels (torch.Tensor): ground truth labels.
        preds (torch.Tensor): logits (not softmaxed yet).
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
        binary (bool): whether doing binary classification or multi-class classification.
    """
    wandb_log[metric_prefix + 'loss'] = loss
    
    if not binary: # multi-class    
        # get probability
        # if(model_type=='SVM'):
        #     preds = torch.softmax(preds, axis = 1)
        # else:
        preds = torch.softmax(preds, axis = 1)

        # ROC AUC
        try:
            wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds, multi_class = 'ovr')
        except Exception:
            wandb_log[metric_prefix + 'auc'] = -1

        # get class prediction
        preds = preds.argmax(axis = 1)
        
        metrics_no_avg = {'accu' : accuracy_score, 'mcc' : matthews_corrcoef, 'time' : (time.time() - current_time), 'memory_usage': memory_info[0]}
        # accuracy and mcc
        for metric_name, metric_func in metrics_no_avg.items():
            if(metric_name=='time' or metric_name=='memory_usage'):
                metric = metric_func
            else:
                metric = metric_func(labels, preds)
            wandb_log[metric_prefix + metric_name] = metric

        # precision, recall, f1 score
        for metric_name, metric_func in metrics_with_avg.items():
            metric = metric_func(labels, preds, average = avg, zero_division = 0)
            wandb_log[metric_prefix + metric_name] = metric
    
    else: # binary
        # get probability
        # if(model_type=='SVM'):
        #     preds = torch.softmax(preds, axis = 0)
        # else:
        preds = torch.softmax(preds, axis = 1)
        
        # ROC AUC
        try:
            wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds)
        except Exception:
            wandb_log[metric_prefix + 'auc'] = -1
        preds = preds.argmax(axis = 1)
        # get class prediction
        preds = preds.round()
        metrics_no_avg = {'accu' : accuracy_score, 'mcc' : matthews_corrcoef, 'time' : time.time() - current_time, 'memory_usage': memory_info[0]}
        # accuracy and mcc
        for metric_name, metric_func in metrics_no_avg.items():
            if(metric_name=='time' or metric_name=='memory_usage'):
                metric = metric_func
            else:
                metric = metric_func(labels, preds)
            wandb_log[metric_prefix + metric_name] = metric

        # precision, recall, f1 score
        for metric_name, metric_func in metrics_with_avg.items():
            metric = metric_func(labels, preds, average = avg, zero_division = 0)
            wandb_log[metric_prefix + metric_name] = metric
