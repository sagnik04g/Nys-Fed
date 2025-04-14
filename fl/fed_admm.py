import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

required=True

class ADMM(Optimizer):

    def __init__(self, params, lambda_i=required, y_k=required, alpha=required,rho=required, device=required):
        
        # if rho is not required and rho < 0.0:
        #     raise ValueError("Invalid learning rate: {}".format(rho))

        defaults = dict(rho=rho,alpha=alpha)
        super(ADMM, self).__init__(params, defaults)
        self.y_k=y_k
        self.alpha=alpha
        self.lambda_i=lambda_i
        self.rho = rho
        param_grps_list=list(self.param_groups)
        h_dim=torch.cat([pi.view(-1) for pi in param_grps_list[0]['params']]).shape
        self.device=device
        self.H = torch.zeros(h_dim[0], h_dim[0]).to(device)

    def __setstate__(self, state):
        super(ADMM, self).__setstate__(state)

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
    
    def compute(self, gradloader, model, model_type):
        "Hessian Computation"
        self.model=model
        for group in self.param_groups:
            col = self.H.shape[0]
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
            for j in range(col):
                if j == col-1:
                    self.H[j] = torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[j], group['params'], retain_graph=False)])
                else:
                    self.H[j] = torch.cat([hi.reshape(-1).data for hi in torch.autograd.grad(g[j], group['params'], retain_graph=True)])

        

    def step(self):
        H_alpha_rho = torch.linalg.pinv(self.H + (self.alpha+self.rho)*torch.eye(self.H.shape[0], self.H.shape[0]).to(self.device))
        for group in self.param_groups:
            g = torch.cat([p.grad.view(-1) for p in group['params']])
            w = self.y_k
            ls=0
            for p in group['params']:
                torch.nan_to_num_(p.data, nan=1e-9, posinf=1e-9, neginf=1e-9)
                yi_k = torch.mm(H_alpha_rho , (g.view(-1,1) - self.lambda_i + self.rho * w.view(-1,1)))
                # self.yi_k=yi_k
                vp=yi_k[ls:ls+torch.numel(p)].view(p.shape)
                ls += torch.numel(p)
                p.grad = vp