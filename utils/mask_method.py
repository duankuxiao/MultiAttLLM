import torch


def _make_generator(device, seed):
    g = torch.Generator()
    try:
        g = torch.Generator(device=device)
    except TypeError:
        pass
    g.manual_seed(seed)
    return g

@torch.no_grad()
def generate_mar_mask(
    X: torch.Tensor,               # [B,T,D]
    f_dim: int,                    # 最后 f_dim 列全部做 MAR 目标
    always_obs: int | list = 4,        # 永久观测的“前几列”：
                                   #   - int: 前 always_obs 列（0..always_obs-1）
                                   #   - list[int]: 直接提供永久观测列的索引（可不局限于“前几列”）
    obs_rate: float = 0.5,     # 中间列中按比例选择“也做 MAR 的列”；其余中间列将“永久观测”
    missing_rate=0.2,              # 目标缺失率：标量或长度==目标列数的向量
    seed: int = 4213,
    max_iter: int = 30,
    eps: float = 1e-8,
):
    """
    返回:
      X_miss: [B,T,D] 置缺后的数据 (NaN=缺失)
      mask  : [B,T,D] 观测指示 (1=观测, 0=缺失)
      info  : dict  索引与达成缺失率统计
    说明:
      - “严格 MAR”：所有被造缺的目标列，其缺失概率仅依赖“永久观测列”（predictors）。
      - 目标列 = 最后 f_dim 列 ∪ 中间段里按 obs_rate_mid 选出的列。
      - 永久观测列 = 手动指定的 always_obs 列 ∪ 中间段里未被选中的列。
    """
    assert X.ndim == 3 or X.ndim == 2, "X 必须是 [B,T,D]"
    if X.ndim == 2:
        X = X.unsqueeze(dim=0)
    device = X.device
    B, T, D = X.shape
    assert 1 <= f_dim <= D, "f_dim ∈ [1, D]"
    # 1) 解析永久观测的“手动列”
    if isinstance(always_obs, int):
        assert 0 <= always_obs <= D, "always_obs (int) 需在 [0, D]"
        base_pred_idx = torch.arange(0, always_obs, device=device)
    else:
        idx = torch.as_tensor(always_obs, device=device)
        if idx.numel() > 0:
            assert torch.all((0 <= idx) & (idx < D)), "always_obs (list) 中存在越界索引"
        base_pred_idx = idx.unique().sort().values

    # 2) 明确尾段目标列（最后 f_dim）
    tail_target_idx = torch.arange(D - f_dim, D, device=device)

    # 3) 拆出“中间段”候选（既不在 base_pred，也不在尾段）
    all_idx = torch.arange(0, D, device=device)
    mid_mask = torch.ones(D, dtype=torch.bool, device=device)
    mid_mask[base_pred_idx] = False
    mid_mask[tail_target_idx] = False
    mid_candidates = all_idx[mid_mask]  # 这些列要被分成：一部分也做 MAR，一部分永久观测

    # 4) 在“中间段”里，按 obs_rate_mid 选择“也做 MAR 的列”
    assert 0.0 <= obs_rate <= 1.0, "obs_rate_mid ∈ [0,1]"
    gen = _make_generator(device, seed)
    if mid_candidates.numel() > 0:
        n_mid_tar = int(round(obs_rate * mid_candidates.numel()))
        if n_mid_tar > 0:
            perm = mid_candidates[torch.randperm(mid_candidates.numel(), generator=gen, device=device)]
            mid_target_idx = perm[:n_mid_tar]
            mid_pred_idx   = perm[n_mid_tar:]
        else:
            mid_target_idx = torch.tensor([], device=device, dtype=torch.long)
            mid_pred_idx   = mid_candidates
    else:
        mid_target_idx = torch.tensor([], device=device, dtype=torch.long)
        mid_pred_idx   = torch.tensor([], device=device, dtype=torch.long)

    # 5) 最终集合
    predictor_idx = torch.cat([base_pred_idx, mid_pred_idx]).unique().sort().values
    target_idx    = torch.cat([tail_target_idx, mid_target_idx]).unique().sort().values

    if predictor_idx.numel() == 0:
        raise ValueError("严格 MAR 需要至少 1 个永久观测（预测器）列。请增加 always_obs 或降低 obs_rate_mid。")
    if target_idx.numel() == 0:
        raise ValueError("未选出任何 MAR 目标列。请增加 f_dim 或增大 obs_rate_mid。")

    # 6) 标准化输入，准备线性打分 S = Xz[:, predictors] @ W → 目标
    N = B * T
    X2d = X.reshape(N, D).float()
    if not torch.isfinite(X2d).all():
        raise ValueError("X 含 NaN/Inf，请先清洗。")

    mu = X2d.mean(dim=0)
    sd = X2d.std(dim=0)
    sd = torch.where(sd < eps, torch.ones_like(sd), sd)
    Xz = (X2d - mu) / sd

    n_pred = predictor_idx.numel()
    n_targ = target_idx.numel()

    # 权重矩阵（小尺度初始化，数值稳健）
    W = torch.randn(n_pred, n_targ, generator=gen, device=device) / (n_pred ** 0.5)
    S = Xz[:, predictor_idx] @ W  # [N, n_targ]

    # 7) 目标缺失率向量
    if isinstance(missing_rate, (list, tuple)):
        mr = torch.tensor(missing_rate, dtype=X2d.dtype, device=device)
    elif isinstance(missing_rate, torch.Tensor):
        mr = missing_rate.to(device=device, dtype=X2d.dtype)
    else:
        mr = torch.full((n_targ,), float(missing_rate), device=device, dtype=X2d.dtype)

    if mr.numel() != n_targ:
        raise ValueError(f"missing_rate 长度应为 {n_targ}（目标列数），当前 {mr.numel()}")
    if not torch.all((mr > 0) & (mr < 1)):
        raise ValueError("missing_rate 中应全部位于 (0,1)")

    # 8) 二分求每个目标列的截距 alpha，使 mean(sigmoid(S + alpha)) = mr
    lo = torch.full((n_targ,), -20.0, device=device)
    hi = torch.full((n_targ,),  20.0, device=device)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p = torch.sigmoid(S + mid)      # [N, n_targ]
        m = p.mean(dim=0)               # [n_targ]
        hi = torch.where(m > mr, mid, hi)
        lo = torch.where(m <= mr, mid, lo)
    alpha = (lo + hi) / 2.0
    miss_prob = torch.sigmoid(S + alpha)  # [N, n_targ]

    # 9) 采样缺失并组装 mask：预测器列永久观测=1；目标列按采样结果
    u = torch.rand(miss_prob.shape, device=device, dtype=miss_prob.dtype, generator=gen)
    mask_t = (u > miss_prob).float()   # [N, n_targ]  1=观测, 0=缺失

    mask2d = torch.ones(N, D, device=device)
    mask2d[:, target_idx] = mask_t
    mask2d[:, predictor_idx] = 1.0     # 永久观测

    # 10) 还原形状并应用
    mask = mask2d.view(B, T, D)
    X_miss = X.clone()
    X_miss[mask == 0] = float("nan")

    info = {
        "mode": "strict_custom",
        "predictor_idx": predictor_idx.tolist(),
        "target_idx": target_idx.tolist(),
        "base_predictor_idx": base_pred_idx.tolist(),
        "mid_target_idx": mid_target_idx.tolist(),
        "mid_predictor_idx": mid_pred_idx.tolist(),
        "tail_target_idx": tail_target_idx.tolist(),
        "achieved_missing_rate_overall": float((mask == 0).float().mean().cpu()),
        "achieved_missing_rate_targets": float((mask[:, :, target_idx] == 0).float().mean().cpu()),
        "n_pred": int(n_pred),
        "n_target": int(n_targ),
    }
    return X_miss


@torch.no_grad()
def generate_mcar_mask(
    X: torch.Tensor,                  # [B, T, D] 完整数据（float，不能含 NaN/Inf）
    missing_rate=0.2,                 # 缺失率：标量 (0,1) 或 1D 向量（见下）
    tail_targets_only: bool = False,  # True: 仅对最后 f_dim 列造缺
    f_dim: int | None = None,         # 当 tail_targets_only=True 时必须提供
    pattern: str = "cell",            # "cell" | "row"
    seed: int = 42,
):
    """
    返回:
      X_miss: NaN 表示缺失，shape=[B,T,D]
      mask:   观测指示 (1=观测, 0=缺失)，shape=[B,T,D]
      info:   元信息（实际缺失率/被造缺列/模式等）

    missing_rate 的用法：
      - 若对“全部特征”造缺：可为标量，或长度 D 的 1D 张量/列表（逐列缺失率）
      - 若仅对“最后 f_dim 列”造缺：可为标量，或长度 f_dim 的 1D 张量/列表
    """
    assert X.ndim == 3 or X.ndim == 2, "X 必须是 [batch, seq, feature]或者[seq, feature]"
    if X.ndim == 2:
        X = X.unsqueeze(dim=0)
    device = X.device
    B, T, D = X.shape
    N = B * T

    X2d = X.reshape(N, D).float()
    if not torch.isfinite(X2d).all():
        raise ValueError("X 含 NaN/Inf，请先清洗。")

    # 选择要造缺的列索引
    if tail_targets_only:
        assert f_dim is not None and 1 <= f_dim <= D, "请提供有效的 f_dim"
        sel_idx = torch.arange(D - f_dim, D, device=device)   # 最后 f_dim 列
    else:
        sel_idx = torch.arange(0, D, device=device)           # 全部列
        f_dim = sel_idx.numel()

    # 统一 missing_rate 为长度=len(sel_idx) 的张量
    if isinstance(missing_rate, (list, tuple)):
        p_vec = torch.tensor(missing_rate, dtype=torch.float32, device=device)
    elif isinstance(missing_rate, torch.Tensor):
        p_vec = missing_rate.to(device=device, dtype=torch.float32)
    else:  # 标量
        p_vec = torch.full((sel_idx.numel(),), float(missing_rate),
                           device=device, dtype=torch.float32)

    if p_vec.numel() != sel_idx.numel():
        raise ValueError(f"missing_rate 长度应为 {sel_idx.numel()}，当前 {p_vec.numel()}")
    if not torch.all((p_vec > 0.0) & (p_vec < 1.0)):
        raise ValueError("missing_rate 需全部在 (0,1) 内。")

    gen = _make_generator(device, seed)

    mask2d = torch.ones(N, D, device=device)  # 默认全观测
    if pattern == "cell":
        # 逐单元格独立伯努利采样：u < p → 缺失
        # 扩展到 [N, C]（C 是被选列数）
        P = p_vec.unsqueeze(0).expand(N, -1)
        U = torch.rand(P.shape, device=device, generator=gen)
        miss = (U < P).float()                     # 1=缺失
        mask_sel = 1.0 - miss                      # 1=观测
        mask2d[:, sel_idx] = mask_sel

    elif pattern == "row":
        # 按时间步（行）采样：u_row < p → 该行在被选列全部缺失
        # 若 p_vec 不是常数，这里采用其均值作为行级 drop 率
        p_row = float(p_vec.mean().clamp(0.0 + 1e-8, 1.0 - 1e-8).cpu())
        Urow = torch.rand((N, 1), device=device, generator=gen)
        keep_row = (Urow >= p_row).float()         # 1=保留行
        mask_sel = keep_row.expand(N, sel_idx.numel())
        mask2d[:, sel_idx] = mask_sel

    else:
        raise ValueError("pattern 只能是 'cell' 或 'row'。")

    # 还原形状并应用
    mask = mask2d.view(B, T, D)
    X_miss = X.clone()
    X_miss[mask == 0] = float("nan")

    # 统计
    info = {
        "pattern": pattern,
        "selected_cols": sel_idx.tolist(),
        "achieved_missing_rate_overall": float((mask == 0).float().mean().cpu()),
        "achieved_missing_rate_selected": float((mask[:, :, sel_idx] == 0).float().mean().cpu()),
    }
    return X_miss


@torch.no_grad()
def generate_rdo_mask(
    X: torch.Tensor,                  # [B, T, D] 完整数据（float）
    row_drop_rate: float = 0.3,       # 行丢失比例（被选列在这些时间步全部缺失），(0,1)
    tail_targets_only: bool = False,  # True: 仅对“最后 f_dim 列”造缺
    f_dim: int | None = None,         # 当 tail_targets_only=True 时必须提供
    mode: str = "bernoulli",          # "bernoulli" | "block"
    share_across_batch: bool = True,  # True: 所有 batch 共享相同行丢失；False: 各样本独立
    mean_off_len: int = 8,            # mode="block" 时，平均缺口长度（时间步）
    seed: int = 42,
):
    """
    返回:
      X_miss: 置缺后的数据 (NaN 表示缺失)，shape=[B,T,D]
      mask:   观测指示 (1=观测, 0=缺失)，shape=[B,T,D]
      info:   元信息（实际缺失率/被造缺列/模式等）

    说明：
      - 这是“整行 drop”：在被选列上，同一时间步要么全观测、要么全缺失。
      - 对被选列的**整体单元格缺失率** ≈ row_drop_rate；
        对全矩阵的整体缺失率 ≈ (len(selected_cols)/D) * row_drop_rate。
    """
    assert X.ndim == 3 or X.ndim == 2, "X 必须是 [batch, seq, feature]或者[seq, feature]"
    if X.ndim == 2:
        X = X.unsqueeze(dim=0)
    assert 0 < row_drop_rate < 1, "row_drop_rate 应在 (0,1)"
    device = X.device
    B, T, D = X.shape

    # 选择造缺列
    if tail_targets_only:
        assert f_dim is not None and 1 <= f_dim <= D, "请提供有效的 f_dim"
        sel_idx = torch.arange(D - f_dim, D, device=device)   # 最后 f_dim 列
    else:
        sel_idx = torch.arange(0, D, device=device)           # 全部列
        f_dim = sel_idx.numel()

    gen = _make_generator(device, seed)

    # ---------- 生成“行级 keep 向量” keep_row: 1=保留，0=丢失 ----------
    if mode == "bernoulli":
        # 按行独立伯努利：u < r → 丢失
        if share_across_batch:
            U = torch.rand((T,), device=device, generator=gen)
            keep_row = (U >= row_drop_rate).float().view(1, T, 1).expand(B, T, 1)  # [B,T,1]
        else:
            U = torch.rand((B, T), device=device, generator=gen)
            keep_row = (U >= row_drop_rate).float().unsqueeze(-1)                   # [B,T,1]

    elif mode == "block":
        # 用二状态马尔可夫链生成“开/关”序列（关=丢失，开=观测）
        # 设 π_off = row_drop_rate，平均“关段”长度 E[L_off]=mean_off_len
        # 转移概率： off->on = a = 1/mean_off_len
        #           on->off = b = a * π_off / (1-π_off)
        assert mean_off_len >= 1, "mean_off_len 应 ≥ 1"
        pi_off = float(row_drop_rate)
        a = 1.0 / float(mean_off_len)
        b = a * pi_off / max(1e-8, (1.0 - pi_off))
        a = min(max(a, 1e-6), 1.0)  # 数值裁剪
        b = min(max(b, 1e-6), 1.0)

        def _one_chain_T(T, gen_local):
            # 初始状态按稳态概率 π_off 采样：1=off(丢失), 0=on(观测)
            s = torch.rand((), device=device, generator=gen_local) < pi_off
            out = torch.empty((T,), dtype=torch.float32, device=device)
            for t in range(T):
                out[t] = 0.0 if s == 0 else 1.0   # 记录 off=1（缺失）
                if s == 1:   # off
                    # 以 a 概率转到 on
                    s = 0 if (torch.rand((), device=device, generator=gen_local) < a) else 1
                else:        # on
                    # 以 b 概率转到 off
                    s = 1 if (torch.rand((), device=device, generator=gen_local) < b) else 0
            # keep_row = 1 - off
            return 1.0 - out

        if share_across_batch:
            keep = _one_chain_T(T, gen)
            keep_row = keep.view(1, T, 1).expand(B, T, 1)  # [B,T,1]
        else:
            # 为每个样本独立生成
            keep_list = []
            # （为了复现性，这里用同一个 gen 也能得到确定性；如需更强隔离，可为每个 b 变更种子）
            for _ in range(B):
                keep_list.append(_one_chain_T(T, gen).view(1, T, 1))
            keep_row = torch.cat(keep_list, dim=0)         # [B,T,1]
    else:
        raise ValueError("mode 只能是 'bernoulli' 或 'block'。")

    # ---------- 构造 mask 并应用 ----------
    mask = torch.ones((B, T, D), device=device)
    # 仅对被选列套用 keep_row；非被选列保持观测=1
    mask[:, :, sel_idx] = keep_row.expand(B, T, sel_idx.numel())

    X_miss = X.clone()
    X_miss[mask == 0] = float("nan")

    # 统计
    info = {
        "mode": mode,
        "row_drop_rate_target": row_drop_rate,
        "mean_off_len": mean_off_len if mode == "block" else None,
        "share_across_batch": share_across_batch,
        "selected_cols": sel_idx.tolist(),
        "achieved_missing_rate_overall": float((mask == 0).float().mean().cpu()),
        "achieved_missing_rate_selected": float((mask[:, :, sel_idx] == 0).float().mean().cpu()),
        "achieved_row_drop_rate": float((keep_row == 0).float().mean().cpu()),
    }
    return X_miss