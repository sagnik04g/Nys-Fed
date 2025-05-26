import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import numpy as np

required=True

class NSGD(Optimizer):

    def __init__(self, params, col_opt=required, irho=required, col=-1,device=required):

        defaults = dict(irho=irho, col=col)
        super(NSGD, self).__init__(params, defaults)
        self.col_opt=col_opt
        param_grps_list=list(self.param_groups)
        self.h_dim=torch.cat([pi.view(-1) for pi in param_grps_list[0]['params']])
        if col < 0:
            columns = self.col_opt * np.ceil(np.log2(self.h_dim.shape[0]))
            col = np.int32(columns)
        if irho is not required and irho < 0.0:
            raise ValueError("Invalid learning rate: {}".format(irho))
        
        self.h = torch.zeros(col, self.h_dim.shape[0]).to(device)
        self.idx = torch.randperm(self.h_dim.shape[0])[:col]
        self.device=device


    def __setstate__(self, state):
        super(NSGD, self).__setstate__(state)

    def compute(self, gradloader, model):

        """Nystrom-Approximated Curvature Information"""
        for group in self.param_groups:
            col = self.h.shape[0]
            for batch_idx, (inputs, targets) in enumerate(gradloader):
                inputs, targets = inputs.cuda(self.device), targets.cuda(self.device)
                self.model = model
                outputs,_ = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                g = torch.autograd.grad(loss, group['params'], create_graph=True, retain_graph=True)
                g = torch.cat([gi.view(-1) for gi in g])
                for j in range(col):
                    if j == col-1:
                        self.h[j] += torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[self.idx[j]], group['params'], retain_graph=False)])
                    else:
                        self.h[j] += torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[self.idx[j]], group['params'], retain_graph=True)])

    def compute_stochastic(self, inputs,targets, model):

        """Nystrom-Approximated Curvature Information"""
        for group in self.param_groups:
            col = self.h.shape[0]
            inputs, targets = inputs.cuda(self.device), targets.cuda(self.device)
            self.model = model
            outputs,_ = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            g = torch.autograd.grad(loss, group['params'], create_graph=True, retain_graph=True)
            g = torch.cat([gi.view(-1) for gi in g])
            for j in range(col):
                if j == col-1:
                    self.h[j] += torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[self.idx[j]], group['params'], retain_graph=False)])
                else:
                    self.h[j] += torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[self.idx[j]], group['params'], retain_graph=True)])
        


    def step(self,gradloader,stoch):
        """Compute the scaled gradient
        """
        if(stoch!=True):
          self.h = self.h/len(gradloader)
        M = self.h[:,self.idx].to(self.device)
        # torch.cuda.synchronize()
        M_eigenvals=torch.linalg.eigvalsh(M)
        min_eigen=torch.min(M_eigenvals)
        print('M:'+str(M.shape)+'\n C='+str(self.h.shape)+'number of batches_k='+str(len(gradloader)))
        print('min. eigen value: '+str(min_eigen))
        g=torch.cat([p.grad.view(-1) for p in self.model.parameters()])
        torch.nan_to_num_(g.data, nan=1e-5, posinf=1e-5, neginf=1e-5)
        norm_g = torch.Tensor.norm(g)
        self.irho = 1/(torch.sqrt(norm_g)).item()
        if min_eigen.item() == 0 or min_eigen.item() <= 1e-6:
            c_t = 1/(self.irho**0.5)
        else:
            c_t = 0
        ## Changed M matrix, removed -ve eigenvals and made matrix P.S.D; change 1 with 2 to make matrix P.D
        M = M + (2*max(-min_eigen.item(),0)+ c_t)*torch.eye(M.shape[0]).to(self.device)
        M_ev=torch.linalg.eigvalsh(M)
        min_eig=torch.min(M_ev)
        print('min. eigen value: '+str(min_eig))
        # g=torch.cat([p.grad.view(-1) for p in group['params']])
        # group['irho'] = torch.linalg.inv(torch.sqrt(torch.Tensor.norm(g)))
        print('Norm g: '+str(norm_g),'irho='+str(self.irho))
        rnk = torch.linalg.matrix_rank(M)
        U, S, V = torch.svd(M)
        ix = range(0, rnk)
        U = U[:, ix]
        S = torch.sqrt(torch.diag(1./S[ix]))
        print('S: '+str(S.shape))
        self.Z = torch.mm(self.h.t(), torch.mm(U, S))
        # self.Q = group['irho']**2 * torch.mm(self.Z, torch.inverse(torch.eye(rnk).to(device) + group['irho'] * torch.mm(self.Z.t(), self.Z)))
        self.Q = self.irho**2 * torch.mm(self.Z, torch.inverse(torch.eye(rnk).to(self.device) + self.irho * torch.mm(self.Z.t(), self.Z)))
        del M
        for group in self.param_groups:
            g=torch.cat([p.grad.view(-1) for p in group['params']])
            # v_new = group['irho']*g.view(-1,1)-torch.mm(self.Q, torch.mm(self.Z.t(), g.view(-1,1)))
            v_new = self.irho*g.view(-1,1)-torch.mm(self.Q, torch.mm(self.Z.t(), g.view(-1,1)))
            ls=0
            for p in group['params']:
                vp=v_new[ls:ls+torch.numel(p)].view(p.shape)
                ls += torch.numel(p)
                p.grad = vp

