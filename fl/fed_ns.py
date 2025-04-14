import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import numpy as np
from scipy.linalg import sqrtm
required=True


class NewtonSketch(Optimizer):

    def __init__(self, params, m_sketch=required, sketch_mu=required, device=required):
        
        # if rho is not required and rho < 0.0:
        #     raise ValueError("Invalid learning rate: {}".format(rho))

        defaults = dict()
        super(NewtonSketch, self).__init__(params, defaults)
        self.m=int(m_sketch)
        param_grps_list=list(self.param_groups)
        self.h_dim=torch.cat([pi.view(-1) for pi in param_grps_list[0]['params']])
        self.device=device
        self.H = torch.zeros(self.h_dim.shape[0], self.h_dim.shape[0]).to(self.device)
        self.mu = sketch_mu
        

    def __setstate__(self, state):
        super(NewtonSketch, self).__setstate__(state)

    def custom_multi_margin_loss(self, x, y, margin=1.0):
    
        n = x.size(0)
        num_classes = x.size(1)
        loss = torch.zeros(n)
        for i in range(n):
            correct_class_score = x[i, y[i]]
            incorrect_class_losses = torch.relu(margin - correct_class_score + x[i, :])
            # incorrect_class_losses[y[i]] = 0
            loss[i] = torch.sum(incorrect_class_losses)

        return loss

    def compute(self, gradloader,model, model_type, stoch):
        "Hessian Computation"
        
        for group in self.param_groups:
            # for batch_idx, (inputs, targets) in enumerate(gradloader):
            inputs, targets = gradloader.dataset[:]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            #optimizer.zero_grad()
            outputs,_ = model(inputs)
            if(model_type == 'SVM'):
                wght=model.logits.weight.view(-1,1)
                loss = torch.mean(self.custom_multi_margin_loss(outputs, targets)) + (0.1 * torch.sum(wght**2))
            else:
                loss = F.cross_entropy(outputs, targets)
                
            g = torch.autograd.grad(loss, group['params'], create_graph=True, retain_graph=True)
            g = torch.cat([gi.view(-1) for gi in g])
            for j in range(self.h_dim.shape[0]):
                if j == self.h_dim.shape[0]-1:
                    self.H[j] = torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[j], group['params'], retain_graph=False)])
                else:
                    self.H[j] = torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[j], group['params'], retain_graph=True)])

        
        H_sqrt=torch.tensor(sqrtm(self.H.cpu().numpy())).float().to(self.device)
        print('Hessian sqrt shape: '+str(H_sqrt.shape))
        S=torch.randn(self.m,H_sqrt.shape[1]).to(self.device)
        self.sketch_H=torch.mm(S,H_sqrt).to(self.device)
        print('Hessian sketch shape: '+str(self.sketch_H.shape))
        return self.sketch_H
        
    
    # def step(self):
    #     client_hessian =  torch.mm(self.sketch_H.T, self.sketch_H)
    #     ls=0
    #     for group in self.param_groups:
    #         g = torch.cat([p.grad.view(-1) for p in group['params']])
    #         v_new = self.mu * torch.mm(torch.linalg.pinv(client_hessian), g.view(-1,1))
    #         for p in group['params']:
    #             vp=v_new[ls:ls+torch.numel(p)].view(p.shape)
    #             ls += torch.numel(p)
    #             p.grad = vp    