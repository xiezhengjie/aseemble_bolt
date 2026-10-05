1. 修复当前正式代码的初始化与兼容接口
   - 在 `SACPolicy` 和高层 `Discriminator` 构造函数中先调用父类初始化，确保 `self.device`、参数注册和 `.to(device)` 在使用前有效。
   - 为 `SACPolicy` 增加兼容性的 `device` 可选参数，并保留当前 `SACPolicy` 为正式类；提供 `SACAgent` 兼容别名，避免旧测试和外部入口因类名迁移直接导入失败。
   - 在 SAC 更新日志中补充旧调用方需要的 `actor_sac_loss`，值与当前 `actor_loss` 一致，不改变优化逻辑。
   - 为高层 `Discriminator` 增加可选 `device` 参数，保留现有判别器网络和损失计算不变。

2. 修复明确断链的导入和资源路径
   - 将活动脚本和诊断脚本中的 `envs.assemble_mujoco_env` 统一改为 `training.envs.assemble_mujoco_env`，保留脚本直接执行所需的仓库根目录路径注入。
   - 将 `record_intervention_data.py` 中错误的 `load_base_policy` 导入改为 `training.model.base.basic_model`。
   - 将判别器测试改为从 `training.policy.discriminator_policy` 导入高层 `Discriminator`，底层 `training.model.gail.discriminator` 继续只提供 `DiscriminatorNN`。
   - 将所有已确认的 `mjcf/`、`urdf/` 和 `/tmp/work` 资源引用统一改为基于仓库根目录的 `assets/mjcf/`、`assets/urdf/` 路径；同时修正相应默认模型/数据路径中已确认的旧目录名。
   - 清理 `camera_calibration.py` 的 trailing whitespace；对其中没有仓库内实现的 `solvepnp`、`src.mujoco_viewer` 保留为外部依赖问题，不伪造替代模块。

3. 恢复仍有生产调用的检查点兼容函数
   - 将历史实现中结构推断逻辑以最小范围恢复到 `training/common/checkpoint.py`，提供 `cfg_from_state_dict`，使 `record_intervention_data.py` 在缺少 `policy_cfg.json` 时仍能按现有权重推断配置。
   - 不恢复旧 `current_frame` 或整套旧 `ResidualSACAgent`，因为当前代码使用 `as_sequence`、`DiffusionPolicy` 与 `SACPolicy`，旧残差类的 GRU/EMA/cond 接口无法通过简单包装兼容。

4. 隔离未迁移的人工残差探针，避免默认测试收集误报
   - 增加最小 pytest 配置，仅收集 `training/tests` 下以 `test_` 开头的真实单元测试，排除 `*_test.py` 人工仿真探针和 `training/scripts/test_key_listener.py`。
   - 对仍引用不存在 `ResidualSACAgent` 的旧探针保留现状并在代码/计划结果中明确其属于尚未迁移的旧架构入口；不添加 `ResidualSACAgent = SACPolicy` 这种会在运行时产生更隐蔽错误的假兼容层。
   - 将 `test_train_utils.py` 中与当前 `ResidualDataCollector.predict_action()` 接口不一致的 mock 断言调整为当前生产接口，测试仍验证动作合成和 warmup 行为。

5. 验证并收敛修改范围
   - 运行 `python3 -m compileall -q training`。
   - 运行 pytest 收集，确认不再因旧包路径、错误判别器导入、`SACAgent`、人工探针或缺失的仓库模块而在 collection 阶段失败。
   - 运行可用的 SAC、判别器、训练工具、策略更新和 diffusion 单元测试；若环境仍缺少 requirements 中已声明但未安装的依赖，记录为环境验证阻塞，不改动算法代码或随意增加依赖。
   - 最后运行 `git diff --check`，仅保留与上述断链修复直接相关的最小修改。