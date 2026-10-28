# 连接器训练数据时序检查

检查数据集 `/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008` 的全部 40 条 episode、40567 帧，并对照采集实现和 SFT loader。检查依据为保存的 host 时间戳与索引，不涉及重新驱动机器人。

## 记录与 chunk 索引

全部帧的 frame_index、episode_index、global index、名义 timestamp 连续且对应 metadata。action.control_cycle_sequence 无缺失或反转，action.command_timestamp 严格递增。所有 state 时间戳与主相机 capture 时间一致且位于保存的插值 bracket 内；所有 action 均在双路图像 ready 之后。state 与 action 没有非有限值。SFT loader 对第 t 帧 observation 取 action[t:t+32]，episode 尾部重复最后一条 action，不跨 episode，无额外帧偏移。

`timestamp` 是 frame_index/30 的名义网格；物理时序应使用 observation.image_timestamp 和 action.command_timestamp。

## 常见配对延迟

主相机 capture / 对齐后的 state 到 action 的延迟中位数 60.48 ms，p95 81.81 ms。双路图像 ready 到 action 中位数 19.27 ms。腕相机相对主相机的有符号 capture 时间差中位数 +5.53 ms，范围 -13.10 至 +28.21 ms。该延迟来自采集器选择图像 ready 后的未使用控制命令，因此不是额外丢掉了两条 action。

episode 0 的首帧 observation 对应的 action[0] 在主相机 capture 后 58.83 ms 下发；action[19] 在该首帧 capture 后 692.89 ms 下发。这个首 chunk 没有检测到控制时序异常。

## 确认的异常

episode 13 的 frame 692/693 控制周期为 43.41/22.42 ms；frame 692 重复前一组 observation。随后主相机/state 到 action 延迟从约 56 ms 增到约 89 ms。该 episode 有 452 帧超出 image_ready_to_command=40 ms，位置在 692 至 1153，最大 62.14 ms。

episode 36 的 frame 948/949 控制周期为 73.16/0.85 ms；frame 949 重复 observation。该 episode 有 12 帧超出上述 ready-to-command 阈值，位置在 948 至 960，最大 55.18 ms。

全数据共 464 帧超出 ready-to-command 阈值，占 1.144%；全部发生在上述两条 episode。其余 38 条通过现有 check_alignment.py 的完整检查。全数据有 32 条重复双路图像记录，约 0.079%，采集器通过复用 observation 保留连续 action。

## 与在线推理的差异

训练使用主相机 capture 时刻插值的 state；在线 flexiv_remote_robot.snapshot 使用当前 measured state 和较早的相机图像。20000 步 trace 主相机帧龄中位数 43.77 ms，腕相机 51.39 ms。因此训练与在线的图像/state 采样关系存在时间错位。该差异应纳入闭环排查；它不能单独解释同一数据首帧输入下模型仍与示范首个 chunk 不一致。

## 判断

没有发现全数据 action 错行、乱序或训练 chunk 固定错一帧的问题。已确认两条 episode 存在局部到持续的配对延迟异常，以及训练与在线 state 采样时刻不同。episode 0 的静止起步偏差不发生在上述异常帧，现有证据不足以将首个预测方向问题归因于这些时序异常。

指标详见 dataset_timing_summary.json；逐 episode 指标详见 episode_timing_details.json；图见 timing_episodes_0_13_36.png。原始数据未修改。
