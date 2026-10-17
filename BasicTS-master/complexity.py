# import torch
# import torch.nn as nn
# import numpy as np
# from torch.profiler import profile, ProfilerActivity
# from pdebench.models.Test_model.Models import Model as StarSolver
#
#
# def get_grid(shape, device):
#     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#     return torch.cat((gridx, gridy), dim=-1).to(device)
#
# initial_step=1
# in_channels=1
# hid_S=16
# T_out=None
# hid_T=128
# embed=2
# spatial_d=2
# freq_d=4
# freq_mlp=1
# freq_use_star=False
# freq_embed=None
# star_d=4
# mlp_ratio=5
# modes=12
# width=20
# drop_path=0.1
# model = StarSolver(T_in=initial_step, C=in_channels, T_out=T_out, hid_S=hid_S, hid_T=hid_T, embed=embed, spatial_d=spatial_d, \
#                    freq_d=freq_d, freq_mlp=freq_mlp, freq_use_star=freq_use_star, freq_embed=freq_embed, modes=modes, width=width, \
#                    star_d=star_d, mlp_ratio=mlp_ratio, drop_path=drop_path).cuda()
# inp = torch.randn(50, 64, 64, 1, 1,).cuda()  # 根据实际输入调整
# grid = get_grid([50, 64, 64], 'cuda:0')
#
# # # 统计 FLOPs
# # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
# #     with torch.no_grad():
# #         model(input_tensor, grid)
# #
# # # 打印结果
# # print(prof.key_averages().table(sort_by="flops", row_limit=10))
# # torch.cuda.reset_peak_memory_stats()
# # with torch.no_grad():
# #     output = model(input_tensor, grid)
# #
# # print(f"Peak memory usage (MB): {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
#
# # 启动 profiler，并启用 FLOPs 统计
#
# from torch.profiler import profile, ProfilerActivity
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     record_shapes=True,
#     with_flops=True  # 关键参数：开启 FLOPs 统计
# ) as prof:
#     with torch.no_grad():
#         model(inp, grid)
#
# # 打印 FLOPs 最多的操作
# print(prof.key_averages().table(sort_by="flops", row_limit=10))
#
# # 打印模型总参数量
# print('Model Parameters:', sum(p.numel() for p in model.parameters()))
#
# # 统计模型总 FLOPs
# total_flops = sum([evt.flops for evt in prof.key_averages() if evt.flops is not None])
# print(f"Total model FLOPs (per batch): {total_flops / (1e9 * inp.shape[0]):.2f} GFLOPs")
#
# # 额外测量显存占用
# # torch.cuda.reset_peak_memory_stats()
# # with torch.no_grad():
# #     output = model(inp, grid)
#
# # print(f"Peak memory usage (MB): {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
import thop  # 修改处：直接导入包



class IEEEModelAnalyzer:
    def __init__(self, model, input_sample_tuple, device='cuda'):
        """
        model: 待测模型
        input_sample_tuple: 单个样本的输入元组 (Batch Size 必须为 1)
        device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.model = copy.deepcopy(model).to(self.device)
        self.model.eval()

        self.inputs = []
        for t in input_sample_tuple:
            self.inputs.append(t.to(self.device))
        self.inputs = tuple(self.inputs)

        self.hooks = []
        self.layer_stats = []
        self.byte_per_elem = 4

    def _register_mem_hook(self):
        def hook_fn(module, inp, out):
            in_vol = sum([t.numel() for t in inp if isinstance(t, torch.Tensor)])
            if isinstance(out, torch.Tensor):
                out_vol = out.numel()
            elif isinstance(out, (tuple, list)):
                out_vol = sum([t.numel() for t in out if isinstance(t, torch.Tensor)])
            else:
                out_vol = 0
            param_vol = sum([p.numel() for p in module.parameters()])
            total_access = (in_vol + param_vol + out_vol) * self.byte_per_elem

            self.layer_stats.append({
                "Layer": module.__class__.__name__,
                "Activations": out_vol * self.byte_per_elem,
                "Memory_Access_Bytes": total_access
            })

        for name, m in self.model.named_modules():
            if len(list(m.children())) == 0 and len(list(m.parameters())) > 0:
                self.hooks.append(m.register_forward_hook(hook_fn))

    def _measure_latency(self, repetitions=100, warmup=20):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((repetitions, 1))

        with torch.no_grad():
            for _ in range(warmup):
                self.model(*self.inputs)

        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                self.model(*self.inputs)
                ender.record()
                torch.cuda.synchronize()
                timings[rep] = starter.elapsed_time(ender)

        return np.mean(timings), np.std(timings)

    def analyze(self):
        print(f"--- Starting IEEE Standard Profiling (Device: {self.device}) ---")

        # 1. 计算 FLOPs (使用 thop)
        # 我们只取 macs，忽略 thop 返回的 params (因为它可能不准)
        macs, _ = thop.profile(self.model, inputs=self.inputs, verbose=False)
        flops = 2 * macs

        # 2. 计算 真实的 参数量 (使用 PyTorch 原生方法) <--- 关键修改
        # 统计所有需要梯度的参数 + 这里的 buffer (如 BatchNorm 的 mean/var 通常不算参数，只算 weight/bias)
        real_params = sum(p.numel() for p in self.model.parameters())

        # 3. 计算访存量 (Memory Access)
        self._register_mem_hook()
        with torch.no_grad():
            self.model(*self.inputs)
        for h in self.hooks: h.remove()

        df_layers = pd.DataFrame(self.layer_stats)
        total_mem_access = df_layers["Memory_Access_Bytes"].sum()
        total_activations = df_layers["Activations"].sum()

        # 4. 计算算术强度
        intensity = flops / (total_mem_access + 1e-9)

        # 5. 测量延迟
        mean_lat, std_lat = self._measure_latency()

        metrics = {
            "Parameters (M)": real_params ,  # 使用真实参数量
            "FLOPs (M)": flops / 1e6,
            "MACs (G)": macs / 1e9,
            "Memory Access (MB)": total_mem_access / 1e6,
            "Activations (MB)": total_activations / 1e6,
            "Arithmetic Intensity (FLOPs/Byte)": intensity,
            "Latency (ms)": f"{mean_lat:.4f} ± {std_lat:.4f}",
            "Throughput (Samples/s)": 1000 / mean_lat
        }

        return metrics


# ==========================================
# 测试部分
# ==========================================

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # ---------------- basic config ----------------
    B = 1
    L = 12
    N = 207
    C_in = 2

    # ---------------- random inputs ----------------
    history_data = torch.randn(B, L, N, C_in, device=device)
    future_data = torch.randn(B, L, N, C_in, device=device)

    dist = torch.rand(B, N, N, device=device)  # 距离矩阵
    sigma = torch.rand(B, device=device) + 1e-3  # 防止除 0

    adj_sc = torch.rand(B, 2, 2, device=device)  # 占位即可

    batch_seen = torch.zeros(1).to(device)

    num_nodes = 207
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # 1) 与截图一致的超参数
    # -------------------------
    gcn_depth = 2
    num_nodes = 207
    dropout = 0.3
    subgraph_size = 20
    node_dim = 40
    middle_dim = 2
    seq_length = 12
    in_dim = 2
    list_weight = [0.05, 0.95, 0.95]
    tanhalpha = 3
    cl_decay_steps = 4000
    rnn_size = 64
    hyperGNN_dim = 16


    # -------------------------
    # 2) predefined_A: list of 2 tensors, each (207, 207)
    # 这里用“随机 + 对称 + 加自环 + 行归一化”，更像真实图结构
    # -------------------------
    def make_predefined_A(n: int, device: str):
        A = torch.rand(n, n, device=device)
        A = (A + A.t()) / 2.0  # 对称
        A.fill_diagonal_(1.0)  # 自环
        A = A / (A.sum(dim=1, keepdim=True) + 1e-12)  # 行归一化（随机游走风格）
        return A


    A0 = make_predefined_A(num_nodes, device)
    A1 = make_predefined_A(num_nodes, device)

    predefined_A = [A0, A1]  # ✅ 与截图一致：list 长度=2，每个 (207,207)

    # -------------------------
    # 3) 打包 kwargs（推荐）
    # -------------------------
    dgcrn_kwargs = dict(
        gcn_depth=gcn_depth,
        num_nodes=num_nodes,
        predefined_A=predefined_A,
        dropout=dropout,
        subgraph_size=subgraph_size,
        node_dim=node_dim,
        middle_dim=middle_dim,
        seq_length=seq_length,
        in_dim=in_dim,
        list_weight=list_weight,
        tanhalpha=tanhalpha,
        cl_decay_steps=cl_decay_steps,
        rnn_size=rnn_size,
        hyperGNN_dim=hyperGNN_dim,
    )
    from baselines.DGCRN.arch.dgcrn_arch import DGCRN

    model = DGCRN(
        **dgcrn_kwargs
    ).to(device)




    analyzer = IEEEModelAnalyzer(model, (
        history_data,
        future_data,
        batch_seen,
        torch.ones(1).to(device),
        torch.tensor(True).to(device),
        dist,
        sigma,
        adj_sc), device=device)
    results = analyzer.analyze()

    print("\n" + "=" * 50)
    print("   Performance Metrics (Single Sample Inference)")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, str):
            print(f"{k:<35} : {v}")
        else:
            print(f"{k:<35} : {v:.4f}")
    print("=" * 50)