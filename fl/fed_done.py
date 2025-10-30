import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

required=True

class DONE(Optimizer):

    def __init__(self, params, d_i_previous=required, alpha=required, device=required):
        
        # if rho is not required and rho < 0.0:
        #     raise ValueError("Invalid learning rate: {}".format(rho))

        defaults = dict(alpha=alpha)
        super(DONE, self).__init__(params, defaults)
        
        self.alpha=alpha
        self.d_i_previous = d_i_previous
        param_grps_list=list(self.param_groups)
        h_dim=torch.cat([pi.view(-1) for pi in param_grps_list[0]['params']]).shape
        self.device=device
        self.H = torch.zeros(h_dim[0], h_dim[0]).to(device)

    def __setstate__(self, state):
        super(DONE, self).__setstate__(state)

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
    
    def compute_hessian(self, gradloader, model, model_type):
        "Hessian Computation"
        self.model=model
        for group in self.param_groups:
            col = self.H.shape[0]
            for inputs, targets in gradloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.backends.cudnn.flags(enabled=False):
                    outputs, _ = model(inputs)
                if(model_type == 'SVM'):
                    wght=model.logits.weight.view(-1,1)
                    loss = torch.mean(self.custom_multi_margin_loss(outputs, targets)) + (0.01 * torch.sum(wght**2))        
                else:
                    loss = F.cross_entropy(outputs, targets)
                g = torch.autograd.grad(loss, group['params'], create_graph=True, retain_graph=True)
                g = torch.cat([gi.view(-1) for gi in g])
                for j in range(col):
                    if j == col-1:
                        self.H[j] += torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[j], group['params'], retain_graph=False)])
                    else:
                        self.H[j] += torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[j], group['params'], retain_graph=True)])
            self.H=self.H/len(gradloader)
            #max_alpha = 2/torch.max(torch.linalg.eigvalsh(self.H))
            #print(max_alpha)
            #if(max_alpha < self.alpha):
            #    self.alpha = max_alpha - 0.5 
            self.H_alpha = (torch.eye(*self.H.shape).to(self.device)-self.alpha*self.H).to(self.device)
            self.d_i_k = torch.mm(self.H_alpha , self.d_i_previous.view(-1,1)) - self.alpha*g.view(-1,1)
            del self.H
            del self.d_i_previous
            return self.d_i_k

    def step(self):
        
        for group in self.param_groups:
            g = torch.cat([p.grad.view(-1) for p in group['params']])
            ls=0
            for p in group['params']:
                torch.nan_to_num_(p.data, nan=1e-9, posinf=1e-9, neginf=1e-9)
                vp = -self.d_i_k[ls:ls+torch.numel(p)].view(p.shape)  ## negation since w=w+(n*d_i)
                ls += torch.numel(p)
                p.grad = vp
