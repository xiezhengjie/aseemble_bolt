# 插装任务残差SAC-GAIL算法设计定稿

# 插装任务：冻结基座 + 残差 SAC + GAIL 算法设计定稿

> **标签约定：反向标签（Shen 同款）** ——D_φ 输出"该样本来自当前策略（生成）"的概率。
> 本版修正记录：① 奖励公式与反标签约定显式锁死，补写 D 损失函数；② 更正"Kulkarni 有判别器"的错误（其无判别器，用稀疏环境奖励）；③ 消除 buffer_e 是否 FIFO 的矛盾（buffer_g 窗口 FIFO，buffer_e 固定）；④ 补 R_succ 进入 Bellman target 的写法与终止吸收态处理；⑤ "下界估计"改为"上界估计"。
> 本方案**不采用 SILfD 成功轨迹注入**。

---

## 0. 标签约定（先锁死，全文档统一，实现前必读）

- **D_φ(s, a) = σ(z) 表示"该 (s,a) 来自当前策略（生成样本）"的概率**，z 为最后一层线性 logit。
- 训练标签：**生成样本（来自 buffer_g）→ 标签 1；专家样本（来自 buffer_e）→ 标签 0**。
- 判别器损失：

```
L_D = − E_{(s,a)~buffer_g}[ log D_φ(s,a) ] − E_{(s,a)~buffer_e}[ log(1 − D_φ(s,a)) ]
```

- 奖励：**r̃ = −log D_φ(s, a^e) = softplus(−z)** 。
  - 样本越像专家 → D_φ→0 → r̃→大（上界由 z clamp 控制）；越像当前策略 → D_φ→1 → r̃→0。方向正确。
  - 与标准标签写法的关系：令 D_GAIL = 1 − D_φ（标准约定下"P(来自专家)"），则 r̃ = −log D_φ ≡ −log(1 − D_GAIL)，与 BeTAIL Eq.6 完全同一件事，只是记号不同（Shen 论文即此写法）。
- **术语警告**：代码与文档中避免"正样本/负样本"这种 GAN 习惯叫法（反标签下含义易混），统一称"**专家样本**（buffer_e，标签 0）"与"**生成样本**（buffer_g，标签 1）"。
- 数值实现：永远从 logit 直算 `r̃ = softplus(−z)`（PyTorch `F.softplus(-z)`），**不要先 sigmoid 再取 log**（饱和后 ±inf/NaN、梯度消失）。z clamp 到 ±20 → r̃ ∈ [≈0, 20]。

---

## 1. 总体架构

### 1.1 组件与角色

- **冻结 GRU（全局唯一）**

  - BC 阶段训练，RL 阶段只推理；输入状态历史，输出 h。
  - RL 阶段一旦冻结，永远不再训练。
- **BC 基座策略**

  - 输入 `[s_t; h_norm]`，其中 `h_norm = LayerNorm(h_t)`（带仿射）。
  - LayerNorm 必须在 BC 阶段与 GRU + BC 头一起训练；RL 阶段随整个基座冻结。
  - 输出基座动作 `a_base = BC(s_t, h_t)`。
- **残差 SAC**

  - Actor：输入 `(s, h)`，输出有界残差 `ã = tanh(θ) × ε`，`ã ∈ [−ε, ε]`。
  - Critic：学 `Q(s, h, ã)`（残差动作视图）。
  - 熵 / log-prob 全部算在 ã 上（BeTAIL 原文："only the entropy of the residual policy"）。
  - SAC 内部只用残差动作，不直接学环境动作。
- **GAIL 判别器 D_φ（反标签，见第 0 节）**

  - 输入 `(s, a^e_执行)`，**不要 h**。
  - Markovian 判别；喂 h 会引发捷径学习（专家数据残差≈0 → D 退化为"残差≈0 即专家"，压死残差策略）。
- **执行动作**

  - `a^e = clip(BC(s,h) + ã, −1, 1)`，`a^e ∈ [−1,1]`。
  - 环境执行与判别器奖励评估都用 a<sup>e；**buffer 存 clip 之后的 a</sup>e**。
- **三个缓冲区**

  - `buffer_r`：SAC 用，完整历史 rollout，~10⁶ 量级。
  - `buffer_g`：D 的生成样本池，短 FIFO 窗口，只存最近 N 条 transition。
  - `buffer_e`：D 的专家样本池，固定专家演示，不注入、不淘汰。

### 1.2 训练阶段

- **BC 阶段**：训练 GRU + BC 头 + 带仿射 LayerNorm；得到确定的 h 表示与基座策略。
- **RL 阶段**：冻结整个基座（GRU + BC 头 + LayerNorm）；只训练残差 actor/critic 与 D_φ；buffer 中所有 h 来自冻结 GRU，永不过期。

### 1.3 为什么可行

- GRU 冻结且确定 → h 是状态历史的确定函数 → `(s, h)` 构成马尔可夫增广状态。
- 因此 SAC 用普通 MLP、按转移级采样即可，**不需要** Kulkarni 那种 w=20 序列回放（那是其 actor/critic LSTM 在线训练的需求）。
- 残差分解贯穿整个 SAC：actor/critic 看 ã；环境执行与 D 奖励评估才重构 a^e。

### 1.4 两条铁律

1. **GRU 进入 RL 阶段后永远不再训练**——否则 buffer 里所有 h 立即失效。
2. **LayerNorm 随基座一起冻结**——不能到 RL 阶段才临时加（未训练的漂移源），也不能在 RL 阶段训练它（等价于动 GRU 表示）。

---

## 2. 判别器 D_φ

- **输入 **​ **​`(s, a^e)`​**​ **，不要 h**；h / 增广状态只进 actor 和 critic。
- 依据：BeTAIL（Eq.6/7）、Shen 均为 Markovian 判别器 D(s,a)；BeTAIL 明确把序列级判别留作 future work。**Kulkarni 无判别器**（稀疏环境奖励 −1/+100），仅其"buffer 存策略输出、执行端混合"的结构与本方案同构。
- 损失：见第 0 节（反标签）。正则：**梯度惩罚 10.0 + 熵正则 0.001**（BeTAIL 原配，不可省；无 SILfD 注入后专家集固定且小，正则更重要）。
- BeTAIL 核查结论：原文 Eq.6 的 D 输入是 `(s, â+ã)`（原始状态 + 环境动作），专家样本以原始 (s,a) 直接送入、不过 BeT、不做残差换算；不存在"D 输入 (s̃,ã)、专家动作残差化"的做法。增广状态 s̃=(s,â) 只服务残差策略与 critic。

---

## 3. 动作、Actor、Critic

|模块|输入|动作视图|
| -----------------| ----------| ----------------------------------|
|Actor（残差策略）|(s, h)|输出 ã = tanh(θ)×ε|
|Critic|(s, h, ã)|残差动作；熵/log-prob 全算在 ã 上|
|判别器 D_φ|(s, a^e)|环境动作|

- 两层约束：残差有界（tanh×ε，策略侧）+ 执行前 clip 到 [−1,1]（环境侧）。**不要**用 tanh(BC+ã) 整体压缩（破坏"残差为零即退化为 BC"）。
- buffer 存 clip 后的 a^e；采样时恢复 `ã = a^e − BC(s_t, h_t)`（BC 冻结，零成本）。
- clip 触发时恢复的 ã_有效 = a^e − a_base，是"实际驱动转移的残差"，给 critic 拟合正确；actor loss 用当前策略新采的 ã′，不受影响。
- 环境动作 a^e 只出现在两处：环境执行、D 的奖励评估。Kulkarni 同构：buffer 存 RL 策略输出，阻抗混合只在执行端。
- ε 别贪大（BeTAIL 消融 α=0.05 显著优于 1.0）；BC 基座贴边界时 clip 会截断残差、压扁该步梯度信号，专家数据很少打满边界，风险可控。

---

## 4. 三个缓冲区

|缓冲区|内容|服务对象|关键机制|
| --------| -------------------------------| ----------------------| -----------------------|
|buffer_r|完整历史 rollout，~10⁶|SAC critic/actor|采样时用当前 D 重算奖励|
|buffer_g|最近 N 条 transition，FIFO 窗口|D 的生成样本（标签 1）|保证生成样本不滞后|
|buffer_e|专家演示 transition，固定|D 的专家样本（标签 0）|不注入、不淘汰|

### 4.1 buffer_r 字段（每条 transition）

|字段|类型|说明|
| ------------| ------------| -------------------------------------------------------------|
|s_t, s_{t+1}|float32 向量|归一化后状态：diag(W)⁻¹(p−p_g)、diag(F_max)⁻¹F|
|h_t, h_{t+1}|float32 向量|rollout 时冻结 GRU 本就算出，顺手存（覆盖"h 不另存"的旧口径）|
|a^e_t|float32 向量|clip 后的实际执行动作|
|r_env|float32|环境奖励（如有；判别器奖励不存，采样时现算）|
|done|bool|终止标志（区分成功/超时/失败）|

- 不存 ã、不存 a_base、不存判别器奖励——采样时全部由冻结模块与当前 D 现算，buffer 永不过期。
- 预填充：用冻结 BC 的 rollout 即可；奖励反正重算，D 随机与否无影响。

### 4.2 buffer_g（生成样本，FIFO 窗口）

- 按 transition 存 `(s_t, a^e_t)`，float32；不需要 h、奖励、done。
- 窗口大小 **N = k·S**，S = 每轮 iteration 采集步数 = M 条 episode × T 步；k 推荐 3~5，起步 k=3。
- 典型量级：T=400、M=14 → S=5600 → **N = 3S ≈ 17,000 条**。
- 下界：≥ 每轮 D 消耗的生成样本数（K_d × batch/2）且 ≥ 10<sub>20 条 episode 的量；上界：窗口跨越的迭代数 × 每轮策略漂移（本架构基座冻结、残差限幅，漂移慢，k 可到 5</sub>10）。
- 性质：N 不改变每条数据一生被抽总次数（= 每轮消耗/每轮新增），只调节批内多样性与陈旧度展布。
- 工程实现：做成 buffer_r 最近 N 条的索引窗口，逻辑两池、物理一份数据。

### 4.3 buffer_e（专家样本，固定）

- 按 transition 存 `(s^E, a^E)` 原始对；**专家数据不过 GRU/BC、不做残差换算**。
- 不注入成功自体轨迹，不做 FIFO 淘汰（本方案放弃 SILfD）。
- 专家数据加载时用与在线相同的**状态归一化**（专家采集时未归一化）。
- 专家原始轨迹文件保留在磁盘：若未来想加"专家数据辅助 BC 正则"，BC loss 需要 h，需靠原始序列重放冻结 GRU 重算。

---

## 5. 状态归一化与 BC 输入

- 状态归一化：`diag(W)⁻¹(p−p_g)`、`diag(F_max)⁻¹F`（Lin 2025 式）；buffer 存归一化数据，专家数据同套归一化。
- BC 输入为 `[s_t; h_norm]` 而非单纯 h_t：
  - 若 GRU 输入滞后一拍则必须拼（否则策略对当前状态无反应）；即使不滞后，拼接 = skip connection，保证当前力/位信号无瓶颈直达策略头（GRU 压缩有损）。
  - 先例：Everett（LSTM 隐状态 + 自身状态拼接进 FC）、Kulkarni（actor 输入 = 观测+隐状态）。
- LayerNorm 时机（铁律 2）：BC 阶段联合训练，RL 阶段冻结；**不用 BatchNorm（batch=1 推理时行为分裂）。**
- 诊断：冻结后在专家数据上统计 h 各维 mean/std、|h|>0.95 饱和占比（>10% 回 BC 阶段修）、有效秩。
- 残差 actor/critic 的输入建议统一用 h_norm（三模块同一表示，实现最简）；用原始 h 也不算错。

---

## 6. 奖励与成功奖励

- 判别器奖励：**r̃ = −log D_φ(s, a^e) = softplus(−z)** ，z clamp ±20，r̃ 上界 20；采样时用**当前** D 重算（buffer 存的奖励不作数）。
- 恒正奖励（r̃ ≥ 0）存在"苟活激励"：成功即提前终止的任务里，策略可能拖延不插入。本方案回合为成功提前终止制，故配成功奖励：
  - **R_succ 要 ≥ 成功后放弃的剩余 GAIL 奖励的上界估计**（宁大勿小，小了压不住拖延）。
  - 标定：r̃_max = 20（z clamp 后），R_succ ≥ (T − t_succ)·r̃_max 的保守估计；最省事取固定值 T·r̃_max 打折；或时间依赖形式 R_succ(t) = (T − t)·r̄_recent（精确补偿放弃的部分）。
- 成功奖励只是回报塑形，**不触发**任何 buffer_e 注入（无 SILfD）。
- 放弃 SILfD 的代价：放弃其消融收益（Shen 报告约 +7%），且防拖延完全依赖 R_succ 标定。

---

## 7. 数据流 / 实现清单

1. rollout：`h = 冻结GRU(s 历史)`，`a_base = BC(s, h_norm)`，执行 `a^e = clip(a_base + ã, −1, 1)`。
2. buffer_r 存 `(s_t, s_{t+1}, h_t, h_{t+1}, a^e_t, r_env, done)`；buffer_g 窗口同步获得 `(s_t, a^e_t)`。
3. 采样时（SAC 批）：取已存 h_t → `a_base = BC(s_t, h_t)` → `ã = a^e_t − a_base` → `r̃ = softplus(−z_当前(s_t, a^e_t))`。
4. SAC 更新（含成功奖励与吸收态处理）：

```
y = r̃ + R_succ·1{success} + γ·(1−done)·[ Q_targ(s′,h′,ã′) − α·log f_res(ã′|s′,h′) ]
```

- ã′ 由当前 actor 在 (s′,h′) 处新采；成功终止：y = r̃ + R_succ（不 bootstrap）；超时/失败终止：y = r̃（不 bootstrap）。

5. D 更新（反标签，第 0 节损失）：生成样本 ∼ buffer_g（标签 1），专家样本 ∼ buffer_e（标签 0），1:1；GP 10.0 + 熵正则 0.001。
6. 预填充 buffer_r：冻结 BC 的 rollout。
7. 轮内顺序：采集 → **SAC（用上一轮的 D）→ D**。一轮滞后，GAN 式交替的标准做法，无需处理。

---

## 8. 更新频率与超参数（最终建议）

|项目|建议值|备注|
| ---------------| -----------------------------------| ------------------------------------------------------|
|SAC 更新|**40 次/轮 × batch 1024**|replay ratio ≈ 7.3；可调 40~100，只动次数不动 batch|
|D 更新|**10 次/轮 × batch 1024**，专家/生成 1:1|可调 5~20，由 D 准确率闭环调节|
|buffer_g 窗口|N = 3S ≈ 17,000 条（k=3 起步）|10 次×512 生成样本 ≈ 窗口 30%，曝光健康|
|SAC 学习率|3e-4（actor、critic 同）||
|D 学习率|1e-3|D 网小（[64,64] 或更小），配合正则|
|D 正则|GP 10.0 + 熵正则 0.001|不可省|
|奖励|r̃ = softplus(−z)，z clamp ±20|反标签，上界 20|
|γ / Polyak|0.99 / 0.005||
|目标熵|−dim(ã)（= 动作维度）|SAC 自动调温|
|迭代规模|14 环境 × 400 步 = 5600 步/轮||

**为什么 40:10 而不是 14:20**：critic 同时追自举目标与每轮重算的奖励面，是真瓶颈（步数线性决定追踪速度，batch 只按 √B 降噪声）；D 是监督学习、且策略漂移慢（基座冻结+残差限幅），10 次足够跟踪，20 次只会逼它背下小窗口与小专家集 → 过自信 → 奖励饱和。比例落在 Shen（0.8:1）与 BeTAIL（78:1）之间。

**调节触发器**：

|现象|动作|
| ---------------------------| --------------------------------|
|D 准确率 >95%，奖励普遍≈0|D 减到 5 次，或加大 GP|
|D 准确率 <60%，奖励无区分度|D 加到 15~20 次|
|critic loss 不收敛 / Q 漂移|SAC 加到 70~100 次|
|成功率升但回合变长（拖延）|检查 R_succ 是否盖住 (T−t)·r̄|

---

## 9. 轨迹级 vs 转移级（归因与本系统选择）

- 判别器在两种方案里都逐 (s,a) 点工作；区别只在采样分布，不在 D 的输入。
- Shen 存轨迹是被"每轮清空 + SILfD 轨迹级成功判定"逼出来的；本方案放弃 SILfD，轨迹级存储的必要性消失。
- BeTAIL 存转移是被 off-policy 大 replay 去相关采样逼出来的。
- **本系统全部按 transition**：buffer_g 窗口 FIFO、buffer_e 固定摊平、buffer_r 完整历史。
- 旧的"轨迹级红利"（h 重放重算）已被覆盖：buffer_r 直接存 h_t、h_{t+1}。

## 10. 文献依据对照

|文献|本方案采用的元素|
| ------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------|
|BeTAIL|残差+AIL 总框架；增广状态思想（以冻结 GRU 的 h 替代 s̃=(s,â)）；奖励用当前 D 重算；D 输入 (s, â+ã)；GP 10.0 + 熵 0.001；ε 取小（α=0.05 消融）|
|Shen|三池结构；反标签写法（r = −log D_φ）；iteration 级清空语义（本方案改为 k·S 窗口）；SAC/D 交替更新结构|
|Kulkarni|**无判别器**（稀疏奖励 −1/+100）；λ_t = max(0,(T−t)/T) 回合内移交（94% vs 24% 消融）；buffer 存策略输出、执行端混合；w=20 序列回放（本系统不需要）|
|Lin 2025|状态归一化公式 diag(W)⁻¹(p−p_g)、diag(F_max)⁻¹F；残差输出力/位修正的参考|
|Everett|LSTM 隐状态 + 自身状态拼接先例（支持 BC 输入 [s; h_norm]）|
|Park HACT|action chunking（列为未来改进方向）|
|ARC（"Goyal"文件）|奖励形式对照（GAIL: log D；f-MAX: log-odds = z）；"谁慢补谁"的更新配比原则（其 C 网 10:1）|

‍

‍

对照定稿后，主要未对齐点如下（按严重程度）：

## 高（算法语义偏离）

**1. 状态归一化不是 Lin 公式，且 buffer / D / SAC 口径不一致**

- 定稿 §4.1/§5：`diag(W)⁻¹(p−p_g)`​、`diag(F_max)⁻¹F`，buffer 存归一化后状态；专家与在线同套。
- 现状：环境只给相对位姿 `p−p_g`​ + 原力，**不做** `W/F_max`​ 缩放；训练用 `RunningMeanStd` z-score。
- <span data-type="text" style="color: var(--b3-font-color13);">更严重：</span>`ResidualCollector`​​<span data-type="text" style="color: var(--b3-font-color13);"> 往 </span>`buffer_r`​​<span data-type="text" style="color: var(--b3-font-color13);"> 写的是</span>**原始观测**<span data-type="text" style="color: var(--b3-font-color13);">，策略/</span>`h`​​<span data-type="text" style="color: var(--b3-font-color13);"> 用的是归一化观测；而 </span>`buffer_e_flat`​​<span data-type="text" style="color: var(--b3-font-color13);"> 存的是</span>**归一化专家态**<span data-type="text" style="color: var(--b3-font-color13);">。</span>于是：

  - D：生成样本（原始）vs 专家样本（归一化）——分布不对齐；
  - SAC：`update`​/`recover_residual`​/`disc.predict_rewards`​ 吃的是 buffer 里的原始 `s`​，但 `h`​ 是在归一化 `s` 上算的——与 BC 训练输入也不一致。

**2. BC 的** **​`obs_normalizer`​**​ **与 RL 入口未强制共用**

- BC 目录里有 `obs_normalizer.npz`​，`train_sac_gail_ur5e.py`​ 却用专家数据重新 `RunningMeanStd.update`，不加载 BC 归一化器。
- 基座若在另一套统计下训出，RL 阶段再换一套，等于分布漂移。

## 中（口径可解释，但与定稿字面不符）

**3.**  **​`buffer_r`​**​ **存原始** **​`s`​**​ **，不存定稿说的“归一化后状态”**   
定稿要求 buffer 直接存归一化 `s`​；代码存 raw，只在 act/`h` 时 normalize。

**4.**  **​`r_env`​**​ **存了但未进 SAC 目标 （如何弄的）** <span data-type="text" style="color: var(--b3-font-color13);">
定稿允许存 </span>`r_env`​​<span data-type="text" style="color: var(--b3-font-color13);">、主奖励是 </span>`r̃+R_succ`​​<span data-type="text" style="color: var(--b3-font-color13);">；环境仍有稠密奖励（含成功 bonus≈80），但训练只用 GAIL + </span>`R_succ=4000`​​<span data-type="text" style="color: var(--b3-font-color13);">。语义上接近“纯 GAIL”，与定稿不冲突，但环境奖励完全闲置，和“如有”的工程预期略岔开。</span>

**5.**  **​`done`​**​ **字段**  
定稿写“区分成功/超时/失败”；代码用 `terminated|truncated`​ + **另存** **​`success`​**。bootstrap 语义正确，但不是定稿表里那种单字段枚举。

**6. h 诊断对象**  
定稿 §5：在**专家数据**上查饱和/有效秩；代码在 `buffer_r.h` 在线子采样上做诊断。

## 低（工程/文档细节）

**7.**  **D 的 GP 实现细节 （有无相同的这样做）**   
定稿写 GP 10.0；代码是 R1 风格（对 `sigmoid(z)`​ 的输入梯度），系数带 `0.5`，并对专家/生成各算一次相加。方向对，和 BeTAIL 原文是否逐字一致未核死。

**8. 评估集数**  
入口 `eval_episodes=20`，定稿未硬性规定；冒烟/注释里常写 30。

**9. Actor 网宽**  
定稿未强制 `[512,512]`​（来自旧基准）；D `[64,64]`、ε=0.05、40:10、k=3、R_succ=4000 等与 §8 基本一致。

---

## 已对齐的部分（便于对照）

反标签 D、`softplus(−z)`​、z clamp±20、GP10+熵0.001、冻结 GRU+`h_norm`​、残差 `ã∈[−ε,ε]`​、`a^e=clip(a_base+ã)`​、三池结构、`N=k·S`​、采集→SAC→D、`R_succ` 进 target、成功/失败都不 bootstrap、无 SILfD。

---

**最该先修的**：把“归一化后的 `s`​”统一写入 `buffer_r`​ / 专家池 / D / SAC（并优先加载 BC 的 `obs_normalizer`）；Lin 公式是下一步物理尺度对齐，可在统一 RMS 口径之后再做。需要的话我可以直接按这个优先级改一版。

‍

‍

‍

‍

在 `discriminator.py`​ 的 `update`​ 方法中，**GP（Gradient Penalty，梯度惩罚）**  和 **熵正则** 是**两个关键的稳定训练技巧**。

---

## 1. GP（R1 梯度惩罚）

### 公式

$$
\mathcal{L}_{\text{GP}} = \frac{\lambda_{\text{gp}}}{2} \left( \|\nabla_{(s,a)} \sigma(z_e)\|_2^2 + \|\nabla_{(s,a)} \sigma(z_g)\|_2^2 \right)
$$

其中：

- $z_e, z_g$：判别器对专家样本和生成样本输出的 logit
- $\sigma(\cdot)$：sigmoid 函数
- $\lambda_{\text{gp}}$：惩罚系数（`gp_coef`，默认 10.0）

### 代码实现

```python
gp = _r1_gradient_penalty(z_e, (e_s, e_a), coef=self.gp_coef) \
   + _r1_gradient_penalty(z_g, (g_s, g_a), coef=self.gp_coef)
```

`_r1_gradient_penalty`​ 计算的是 **sigmoid 输出对输入** **$(s,a)$** **的梯度平方和**，即 R1 惩罚。

### 作用

|作用|说明|
| ------| --------------------------------------------|
|**防止梯度爆炸/消失**|限制判别器输入附近的梯度幅值，使训练更稳定|
|**增强 Lipschitz 约束**|类似于 WGAN-GP，保证判别器不会过于陡峭|
|**改善样本质量**|避免判别器对微小输入变化过度敏感|

---

## 2. 熵正则（Entropy Regularization）

### 公式

$$
\mathcal{L}_{\text{ent}} = -\lambda_{\text{ent}} \cdot \left[ H(\sigma(z_e)) + H(\sigma(z_g)) \right]
$$

其中：

- $H(p) = -[p \log p + (1-p) \log(1-p)]$：二值熵
- $\lambda_{\text{ent}}$：熵正则系数（`entropy_coef`）

### 代码实现

```python
ent = _disc_entropy(z_e) + _disc_entropy(z_g)
loss = loss_e + loss_g + gp - self.entropy_coef * ent
```

注意是 **减号**：最大化熵，防止判别器过于自信。

### 作用

|作用|说明|
| ------| ----------------------------------------------|
|**防止过自信**|避免判别器过早达到 $D \approx 0$ 或 $D \approx 1$，保持一定的不确定性|
|**缓解模式崩溃**|防止生成器找到判别器的"盲点"后停止探索|
|**提供有用梯度**|当专家与策略差异较小时，仍能提供非零梯度信号|

---

## 3. 总损失函数

$$
\mathcal{L}_{\text{total}} = \underbrace{\text{BCE}(z_e, 0) + \text{BCE}(z_g, 1)}_{\text{反标签分类损失}} + \underbrace{\mathcal{L}_{\text{GP}}}_{\text{梯度惩罚}} - \underbrace{\lambda_{\text{ent}} \cdot \mathcal{L}_{\text{ent}}}_{\text{熵正则}}
$$

### 反标签约定

- **专家样本**标签为 **0**（不像标准 GAIL 标签为 1）
- **生成样本**标签为 **1**
- 这样设计使得 $z \to -\infty$ 时奖励 $\tilde{r} = \text{softplus}(-z)$ 较大，符合"专家应获得高奖励"的直觉

---

## 4. 监控指标

`update` 返回的日志中包含：

|指标|含义|
| ------| ---------------------------------------------------------------|
|`disc_gp`|梯度惩罚项数值|
|`disc_entropy`|判别器输出熵（越高越不确信）|
|`disc_acc`|判别器分类准确率（应保持在 0.7~0.9 之间，过高说明训练不平衡）|
|`disc_expert_value`|$1 - D_\phi(\text{专家})$，即判别器认为样本来自专家的概率|
|`disc_gen_value`|$D_\phi(\text{生成})$，即判别器认为样本来自策略的概率|

‍

‍

**有无问题？？**
