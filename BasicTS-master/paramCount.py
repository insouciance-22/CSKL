import torch
from thop import profile, clever_format
from baselines.DCRNN.arch.dcrnn_arch import D

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- config ---
B = 1
L_in = 12
L_out = 12
N = 207
C_in = 1

# --- dummy inputs ---
history = torch.randn(B, L_in, N, C_in, device=device)
dist0 = torch.rand(N, N, device=device)
sigma0 = torch.tensor(1.0, device=device)
dist = [dist0]
sigma = [sigma0]

# --- build model ---
model = DCRNN(adj_mx, num_nodes=N, seq_len=L_in, horizon=L_out,
              rnn_units=64, num_rnn_layers=2, input_dim=C_in, output_dim=1,
              max_diffusion_step=2, cl_decay_steps=2000,
              use_curriculum_learning=False).to(device)
model.eval()

# --- params ---
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total params:", total, "Trainable params:", trainable)

# --- thop ---
macs, params = profile(model, inputs=(history, None, None, dist, sigma), verbose=False)
macs, params = clever_format([macs, params], "%.3f")
print("MACs:", macs, "Params(thop):", params)
