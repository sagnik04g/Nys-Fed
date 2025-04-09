import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import numpy as np
import gc

required=True

class NYS_ADMM(Optimizer):

    def __init__(self, params, col_opt=required, lambda_i=required, y_k=required, alpha=required,rho=required, col=-1, device=required):
        
        # if rho is not required and rho < 0.0:
        #     raise ValueError("Invalid learning rate: {}".format(rho))

        defaults = dict(col=col,rho=rho,alpha=alpha)
        super(NYS_ADMM, self).__init__(params, defaults)
        self.y_k=y_k
        self.alpha=alpha
        self.lambda_i=lambda_i
        self.rho = rho
        self.col_opt=col_opt
        param_grps_list=list(self.param_groups)
        h_dim=torch.cat([pi.view(-1) for pi in param_grps_list[0]['params']])
        if col < 0:
            columns = self.col_opt * np.ceil(np.log2(h_dim.shape[0]))
            col = np.int32(columns)
        self.device=device 
        self.h = torch.zeros(col, h_dim.shape[0]).to(device)
        self.idx = torch.randperm(h_dim.shape[0])[:col]
        self.col=col
        

    def __setstate__(self, state):
        super(NYS_ADMM, self).__setstate__(state)

    def compute(self, gradloader, model, model_type):

        # (inputs, targets)  for entire dataset deterministic update:
        X, y = gradloader.dataset[:]
        X, y = X.to(self.device), y.to(self.device)
        """Nystrom-Approximated Curvature Information"""
        
        for group in self.param_groups:
            outputs, _ = model(X)
            if(model_type == 'SVM'):
                weights = model.logits.weight.view(-1,1)
                loss = F.multi_margin_loss(outputs, y) + (0.01 * torch.sum(weights**2))
            else:
                loss = F.cross_entropy(outputs, y)

            g = torch.autograd.grad(loss, group['params'], create_graph=True, retain_graph=True)
            g = torch.cat([gi.view(-1) for gi in g])
            for j in range(self.col):
                if j == self.col-1:
                    self.h[j] = torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[self.idx[j]], group['params'], retain_graph=False)])
                else:
                    self.h[j] = torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[self.idx[j]], group['params'], retain_graph=True)])
            # self.h = self.h/len(X)
            M = self.h[:,self.idx].to(self.device)
            M_eigenvals=torch.linalg.eigvalsh(M)
            min_eigen=torch.min(M_eigenvals)
            torch.nan_to_num_(g.data, nan=1e-5, posinf=1e-5, neginf=1e-5)
            norm_g = torch.Tensor.norm(g)
            self.irho = 1/max(torch.sqrt(norm_g).item(),1)
            if min_eigen.item() == 0 or min_eigen.item() <= 1e-5:
                c_t = 1/(self.irho**0.5)
            else:
                c_t = 0
            ## Changed M matrix, removed -ve eigenvals and made matrix P.S.D; change 1 with 2 to make matrix P.D
            M = M + (2*max(-min_eigen.item(),0)+ c_t)*torch.eye(M.shape[0]).to(self.device)
            M_ev=torch.linalg.eigvalsh(M)
            min_eig=torch.min(M_ev)
            print('min. eigen value: '+str(min_eig))
            print('Norm g: '+str(norm_g),'irho='+str(self.irho))
            rnk = torch.linalg.matrix_rank(M)
            U, S, V = torch.svd(M)
            ix = range(0, rnk)
            U = U[:, ix]
            S = torch.sqrt(torch.diag(1./S[ix]))
            print('S: '+str(S.shape))
            self.Z = torch.mm(self.h.t(), torch.mm(U, S))
            self.Q = self.irho**2 * torch.mm(self.Z, torch.inverse((self.rho+self.alpha)*torch.eye(rnk).to(self.device) + self.irho * torch.mm(self.Z.t(), self.Z)))
            del M

    
    def compute_stochastic(self, loss, model):

        """Nystrom-Approximated Curvature Information"""
        for group in self.param_groups:
            
            g = torch.autograd.grad(loss, group['params'], create_graph=True, retain_graph=True)
            g = torch.cat([gi.view(-1) for gi in g])
            for j in range(self.col):
                if j == self.col-1:
                    self.h[j] = torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[self.idx[j]], group['params'], retain_graph=False)])
                else:
                    self.h[j] = torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[self.idx[j]], group['params'], retain_graph=True)])
            M = self.h[:,self.idx].to(self.device)
            M_eigenvals=torch.linalg.eigvalsh(M)
            min_eigen=torch.min(M_eigenvals)
            torch.nan_to_num_(g.data, nan=1e-5, posinf=1e-5, neginf=1e-5)
            norm_g = torch.Tensor.norm(g)
            self.irho = 1/max(torch.sqrt(norm_g).item(),1)
            if min_eigen.item() == 0 or min_eigen.item() <= 1e-5:
                c_t = 1/(self.irho**0.5)
            else:
                c_t = 0
            ## Changed M matrix, removed -ve eigenvals and made matrix P.S.D; change 1 with 2 to make matrix P.D
            M = M + (2*max(-min_eigen.item(),0)+ c_t)*torch.eye(M.shape[0]).to(self.device)
            M_ev=torch.linalg.eigvalsh(M)
            min_eig=torch.min(M_ev)
            print('min. eigen value: '+str(min_eig))
            print('Norm g: '+str(norm_g),'irho='+str(self.irho))
            rnk = torch.linalg.matrix_rank(M)
            U, S, V = torch.svd(M)
            ix = range(0, rnk)
            U = U[:, ix]
            S = torch.sqrt(torch.diag(1./S[ix]))
            print('S: '+str(S.shape))
            self.Z = torch.mm(self.h.t(), torch.mm(U, S))
            self.Q = self.irho**2 * torch.mm(self.Z, torch.inverse((self.rho+self.alpha)*torch.eye(rnk).to(self.device) + self.irho * torch.mm(self.Z.t(), self.Z)))
            del M
           
        


    def step(self):
                
        """Compute the scaled gradient
        """
        ls=0
        for group in self.param_groups:
            g = torch.cat([p.grad.view(-1) for p in group['params']])
            v_new = self.irho*g.view(-1,1)-torch.mm(self.Q, torch.mm(self.Z.t(), g.view(-1,1)))
            for p in group['params']:
                vp=v_new[ls:ls+torch.numel(p)].view(p.shape)
                ls += torch.numel(p)
                p.grad = vp