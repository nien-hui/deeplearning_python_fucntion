import torch

## the grad_outputs will be the constant

x=torch.rand(2,2,requires_grad=True)
print(x)
y=torch.pow(x,2)
print(y)
z=torch.sum(y*x)
print(z)
z.backward(retain_graph=True)

print(z)
dzdy=torch.autograd.grad(z,y,retain_graph=True)[0]
dzdx=torch.autograd.grad(y,x,grad_outputs=dzdy, retain_graph=True)[0]
print(dzdx)

print(z)
dzdy=torch.autograd.grad(z,y,retain_graph=True)[0].detach()
dzdx=torch.autograd.grad(y,x,grad_outputs=dzdy)[0]
print(dzdx)