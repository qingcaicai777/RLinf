# Flexiv Stage2 按键与操作

这份表格说明 Stage2 推理、人工接管和 episode 结束时的按键操作。在机器人客户端所在的交互式终端中直接按键，无需回车；字母不区分大小写。

当前关节 Stage2 配置会根据左侧执行臂的实测 TCP 位置自动请求 Fine：在机器人基坐标系中，XY 到 `(480, 53) mm` 的平面距离不超过 `15 mm`，且 `Z < 138 mm`，连续满足 3 个 30 Hz 控制周期后触发。请求复用 I 的交接流程，实际切换仍等待合法交接点；每条 episode 只自动请求一次。暂停、人工接管、Home、力清零和 episode 结束后的等待期间不触发。开始下一条 episode 后重新启用判断，手动 I 仍可提前请求切换。

机器人端从两端共用的 inference YAML 读取以下设置；删除 `auto_fine` 或设为 `null` 后恢复只使用手动 I 的方式。

```yaml
auto_fine:
  xy_center_mm: [480.0, 53.0]
  xy_radius_mm: 15.0
  z_below_mm: 138.0
  consecutive_ticks: 3
```

| 按键 | 使用条件 | 说明与操作结果 |
| --- | --- | --- |
| **I** | 模型控制中，服务端支持 Stage2 | 请求切入 Fine 模式，在有效交接点开始按 6 帧调度，并按 `joint_control.fine_motion_limit_scale` 缩小模型关节控制的速度、加速度上限。采集人工修正数据时，先完成此切换。 |
| **Space（空格）** | 模型控制中 | 立即停止下发模型运动目标，后台对两臂调用 SDK Stop；全部停稳后读取位置、恢复保持控制。出现 `Pause complete` 后，才能按 R、H 或 Space；模型继续后台推理。 |
| **R** | 当前 episode 未结束，已暂停且尚未人工接管 | 保持两臂静止、无外力接触，等待力偏置清零完成，再拖动右臂进行人工接管，左臂跟随。 |
| **O** | 已暂停或人工接管中 | 打开左臂夹爪；宽度、速度和力使用 myq 配置。 |
| **C** | 已暂停或人工接管中 | 闭合左臂夹爪；宽度、速度和力使用 myq 配置。 |
| **Space（再次按下）** | 已暂停或人工接管中 | 当前 episode 未结束时恢复模型控制；S/F/D 后直接按 Space，即可从当前位置开始下一条 episode 的粗阶段；H 回 Home 可选。 |
| **H** | 已暂停或人工接管中 | 两臂返回 Home，直接将夹爪目标设为 **0.06 m**，使用 myq 配置的速度和力，然后清零力偏置。完成后仍暂停，按 Space 从粗阶段恢复模型控制及配置限速；到精细区域后重新自动触发 Fine，也可按 I。 |
| **S** | episode 运行中 | 标记成功并保存有效训练窗口，退出人工接管并保持暂停，模型继续后台推理。 |
| **F** | episode 运行中 | 标记失败并保存有效训练窗口，退出人工接管并保持暂停，模型继续后台推理。 |
| **Q** | 客户端运行中 | 停止客户端；当前 episode 未标注时暂存有效窗口，供后续补标签。已通过 S/F 保存的数据保留。 |
| **D** | episode 运行中 | 丢弃当前 episode 的训练数据，退出人工接管并保持暂停，模型继续后台推理；已保存的 episode 保留。之后可按 H 回 Home，或直接按 Space 开始下一条。 |

按以下顺序完成常用操作：

| 操作目标 | 按键顺序 |
| --- | --- |
| 采集人工修正 | I → 等待进入 Fine 模式 → Space → R → 等待清零完成 → 拖动右臂，按 O/C 控制夹爪 |
| 人工修正后继续模型执行 | Space → 等待新预测交接 |
| 暂停检查后继续 | Space → 检查 → Space |
| 返回 Home | Space → H → 等待返回 Home、夹爪打开和清零完成 → Space；若已暂停或接管，直接按 H |
| 从当前位置连续采集 | S/F/D → Space → 粗阶段模型控制 → I → 精细阶段采集 → S/F；重复此流程 |
| 回 Home 后连续采集 | S/F/D → H → 等待回 Home、夹爪打开至 0.06 m 和清零完成 → Space → 粗阶段模型控制 → I → 精细阶段采集 → S/F |
| 丢弃后连续采集 | D → 可选 H 并等待完成 → Space → 粗阶段模型控制 → I → 精细阶段采集 |
| 停止客户端 | 按 Q；若当前数据不保留，先按 D 丢弃，再按 Q 退出 |

返回 Home、夹爪打开和力偏置清零期间，等待完成后再按 Space、R、H 或 O/C。H 回 Home 后直接发送夹爪位置命令，不执行夹爪初始化。暂停、Home 和清零窗口不进入训练；人工接管期间模型持续推理，有效 Fine 窗口可用于训练。S/F/D 后机械臂保持暂停，模型持续后台推理，此期间不采集 RL transition。直接按 Space 即可创建独立的新 episode，从当前位置恢复粗阶段模型控制；也可以先按 H 回 Home，再按 Space。新 episode 清空旧动作与 RTC 上下文，重新请求预测。每条 episode 重新判断自动触发区域；自动切换或手动按 I 后进入精细阶段采集 RL transition。连续采集时保持服务端和客户端运行；更新此功能后先重启两端一次。

启用二值夹爪后，用以下配置选择闭合期间的机械臂行为：

```yaml
gripper_policy:
  binary: true
  threshold: 0.5
  hold_arm_on_close: true
```

| `hold_arm_on_close` | 模型触发闭合时的行为 |
| --- | --- |
| `true`（默认） | 保持机械臂测得的位置，夹爪动作结束后自动请求新预测并恢复模型控制，无需按 Space。 |
| `false` | 夹爪异步闭合，机械臂继续执行模型动作。 |

两种设置均使用配置中的夹爪速度和 50 N 闭合力上限。此开关作用于模型触发的二值闭合；O/C 和 H 保持各自已有控制流程。修改后重启机器人客户端。

采集前确认客户端出现 `robot entered fine phase`。自动检测满足条件时先打印 `Automatic Fine requested`，该日志表示已请求切换；实际进入 Fine 后才能采集有效 Fine 窗口。新 episode 启动推理尚未完成时请求 Fine，会排队保留，并在首个合法粗阶段交接点切入 Fine。若上条 episode 已经通过 S/F 结束，先按 Space 开始新 episode，等待自动切入 Fine 或手动按 I，再按 Space → R；暂停或接管时按 I 会提示未生效。

按 D 丢弃的 episode 不保存、不暂存待标注数据，也不计入训练门槛；结束摘要中的 `collection/discarded` 为 1。按 S/F 保存后检查 `collection/valid_windows` 是否大于 0。`No RL transitions saved` 表示本条没有保存有效窗口，不计入前 50 条 episode；此时查看 `Episode ... scheduling` 的 `fine_requests`、`fine_accepted` 和 `fine_discarded`。若 `fine_requests=0`，检查 I 是否生效及是否完成切换；若 Fine 请求被持续丢弃，检查推理是否超过 6 帧预算。保存目录中的 `.pt` 文件才包含已接纳的 transition。

绝对关节角模式在粗阶段使用 `joint_control` 配置的速度和加速度上限；按 I 后，在实际进入 Fine 的交接点将两者乘以 `joint_control.fine_motion_limit_scale`。当前配置对应粗阶段每关节 0.06 rad/s、0.15 rad/s²，配置比例为三分之一时，Fine 阶段为 0.02 rad/s、0.05 rad/s²。重复按 I 不会重复缩小上限；普通 Space 暂停后恢复仍使用当前阶段的上限；H 回 Home 后再按 Space，以及下一条 episode 回到粗阶段时，均恢复配置值。人工接管与 H 回 Home 使用各自的 myq 控制参数。

```yaml
joint_control:
  max_joint_velocity_rad_s: [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]
  max_joint_acceleration_rad_s2: [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
  fine_motion_limit_scale: 0.3333333333333333
```

`fine_motion_limit_scale` 须为大于 0 且不超过 1 的有限数值；设为 1 时 Fine 使用粗阶段上限。未配置时保持此前的 0.5 行为。

默认启动并打开夹爪时，客户端会等待实测宽度到达配置打开目标 ±1 mm 且夹爪停止运动，再进入首段推理；等待超时则中止启动。`--skip-gripper-init` 或 `open_on_start: false` 仍保留当前配置的行为。

H 后恢复会同步重置机器人调度和服务端为粗阶段，并清空旧动作与 RTC；当前 episode ID 和已收集的数据保留。当前配置 `fine_motion_limit_scale: 0.2` 对应粗阶段 0.06 rad/s、0.15 rad/s²，Fine 阶段 0.012 rad/s、0.03 rad/s²。

## 保存与继续训练

S/F 确认的有效 episode 会先原子保存到 `episode_dir`，再交给独立 Learner；后台训练积压不会阻塞落盘。Learner 每次 Critic 更新之间检查新数据，新 episode 入库后立即保存完整 `latest.pt`，训练期间每约 10 秒在更新边界保存一次，正常关闭时也保存。checkpoint 包含 Actor、Critic、优化器、Replay、更新额度和预热进度。

Learner 异常退出会在服务端明确报错，并将错误写入 `checkpoint_dir/learner_error.log`；进程被强制结束时记录 PID 和退出码。已经落盘的 episode 会在下次启动时重新扫描入库，不会因 Learner 未消费队列而丢失。强制退出仍可能损失最近一次 checkpoint 之后的训练更新；正常关闭会等待已接收的有限更新额度完成，保存后再退出。

恢复时先退出机器人客户端和旧推理服务，再启动服务端，最后启动客户端。两端使用同一份配置，保留原来的 `episode_dir` 和 `checkpoint_dir`。当前旧运行的 `latest.pt` 停在训练前，需要首次使用已核查的完整 `actor_250.pt` 恢复：

```bash
.venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --inference-config examples/embodiment/config/flexiv_joint_absolute_stage2.yaml \
  --host 127.0.0.1 --port 18765 \
  --resume-stage2 results/flexiv_joint_stage2_ckpt1014_20000/checkpoints/actor_250.pt
```

服务端就绪后，在机器人终端运行：

```bash
/home/qccc/miniconda3/envs/flexiv/bin/python \
  examples/embodiment/run_flexiv_online_residual_robot.py \
  --inference-config examples/embodiment/config/flexiv_joint_absolute_stage2.yaml \
  --backend torch --host 127.0.0.1 --port 18765
```

这份 checkpoint 已完成 50 条 episode 的采集门槛和 Critic 预热，恢复 Actor 更新计数 250，继续使用剩余更新额度并接收新数据。机器人仍从粗阶段开始，进入 Fine 才使用恢复后的 Actor。新代码产生更新后的 `latest.pt` 后，今后把 `--resume-stage2` 路径改为 `.../checkpoints/latest.pt` 即可。只指定 `--actor-checkpoint` 不会恢复完整训练状态。

训练门槛使用 Learner 已接收的 episode 计数；删除磁盘 `.pt` 不会撤销内存 Replay 或 checkpoint 中的样本。若希望完全重采并重新训练，请使用新的数据与 checkpoint 目录，并移除恢复参数。
