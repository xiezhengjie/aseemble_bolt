为普通和残差 SAC-GAIL 共用的判别器加入 BeTAIL 风格的预测熵正则，默认系数 `0.001`。

1. 修改 `training/model/gail/discriminator.py`：在构造函数末尾新增 `ent_reg_scale=0.001`，避免破坏现有位置参数；校验系数为有限非负数。拼接专家与策略 logits，计算二元预测熵 `H(p) = -p log p - (1-p) log(1-p)`，以数值稳定、保持梯度的形式实现。总损失为 `原有 expert_BCE + policy_BCE - ent_reg_scale * mean(H)`，保留原来的标签平滑和 BCE 缩放。
2. 在 `config/train_sac_gail.yaml` 和 `config/train_sac_gail_residual.yaml` 的 `discriminator` 下添加 `ent_reg_scale: 0.001`；设置为 `0` 可关闭正则。
3. 判别器返回日志增加 `bce_loss`、`entropy`、`entropy_loss`，现有训练器会自动加上 `disc_` 前缀，便于区分分类损失和熵项。
4. 新增针对判别器的 `unittest`：验证损失公式、关闭正则时退化为原损失、熵项梯度推动非零 logits 向零靠近、极端 logits 下损失及梯度有限，以及默认值和非法参数。
5. 使用本机已有 Python/PyTorch 环境运行新增测试及现有 `test_train_utils`、`test_diffusion_sequence` 回归测试，并验证两份 Hydra 配置可实例化判别器。当前 shell 没有 `python` 命令，会定位现有环境，不安装或改动系统环境。

不修改 SAC 的 Actor/Critic 熵、温度、判别器奖励公式和训练顺序；保留工作区其他已有改动。