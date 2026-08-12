import torch
import torch.nn as nn

# ۱. تعریف معماری شبکه عصبی PINN
class PINN1DHeat(nn.Module):
    def __init__(self):
        super(PINN1DHeat, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# ۲. محاسبه تابع زیان فیزیکی (PDE Loss)
def compute_pde_loss(model, x, t, alpha=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)

    T = model(x, t)

    dT_dt = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dT_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]

    pde_residual = dT_dt - alpha * dT_dx2
    return torch.mean(pde_residual ** 2)

# ۳. تابع آموزش مدل
def train_pinn():
    model = PINN1DHeat()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    N_f = 1000
    N_b = 200
    N_i = 200

    print("شروع آموزش شبکه عصبی PINN...")
    for epoch in range(2001):
        optimizer.zero_grad()

        # تولید نقاط داخل حلقه جهت جلوگیری از اشتراک گراف محاسباتی بین اپوک‌ها
        x_f = torch.rand(N_f, 1, requires_grad=True) * 1.0
        t_f = torch.rand(N_f, 1, requires_grad=True) * 0.5

        t_b = torch.rand(N_b, 1) * 0.5
        x_b0 = torch.zeros(N_b, 1)
        x_bL = torch.ones(N_b, 1)

        x_i = torch.rand(N_i, 1)
        t_i = torch.zeros(N_i, 1)
        T_ic_target = torch.sin(torch.pi * x_i)

        loss_pde = compute_pde_loss(model, x_f, t_f)
        
        T_b0 = model(x_b0, t_b)
        T_bL = model(x_bL, t_b)
        loss_bc = torch.mean(T_b0**2) + torch.mean(T_bL**2)

        T_ic_pred = model(x_i, t_i)
        loss_ic = torch.mean((T_ic_pred - T_ic_target)**2)

        total_loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_ic
        
        total_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss PDE: {loss_pde.item():.6f} | Loss BC: {loss_bc.item():.6f} | Loss IC: {loss_ic.item():.6f}")

    return model

# جلوگیری از اجرای خودکار هنگام import شدن
if __name__ == "__main__":
    trained_pinn = train_pinn()