Flexiv PI0.5 SFT 与 RTC 推理
============================

本页说明如何在 RLinf 中训练双相机、8D state 的 Flexiv PI0.5 contract，并用标准 OpenPI/JAX SFT checkpoint 运行异步 RTC 推理。部署链路只包含普通 speed-field guidance，不调用 APXinf、RL Token、残差 Actor、Critic 或在线训练。

准备机器人主机
--------------

RLinf 安装脚本会创建 OpenPI 环境并安装 RealSense 依赖。``OPENPI_RTC_PATH`` 必须指向已经审计、支持 ``return_raw_actions`` 与普通 speed-field sampling 的 OpenPI checkout；已发布的 ``rlinf-openpi==0.1.1`` wheel 不包含这些 API。Flexiv RDK 仍由机器人主机提供；激活机器人之前，所选环境必须能够执行 ``import flexivrdk``。

.. code-block:: bash

   export OPENPI_RTC_PATH=/absolute/path/to/rtc-compatible/openpi
   bash requirements/install.sh embodied --venv openpi --model openpi --env flexiv
   source switch_env openpi
   python -c "import flexivrdk, pyrealsense2"

训练标准 SFT contract
---------------------

训练 config 读取 Flexiv compact 数据，构造 12 步 chunk，把记录的绝对 TCP target 转换为相对当前样本 measured TCP 的 world-frame delta，并在不启用 RL Token 的情况下训练 7D 输出。

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh flexiv_sft_openpi_pi05

启动前，需要替换 ``examples/sft/config/flexiv_sft_openpi_pi05.yaml`` 中的 ``data.train_data_paths``、``actor.model.model_path`` 和 ``actor.model.openpi_data.norm_stats_path``。RLinf FSDP trainer 保存 PyTorch 训练 checkpoint；下文的 RTC 入口只接受包含 ``params/`` 和匹配 ``assets/`` 的标准 OpenPI/JAX checkpoint。将 RLinf FSDP checkpoint 导出为该格式属于独立步骤，入口脚本不会隐式转换。

配置部署语义
------------

复制 ``examples/embodiment/config/flexiv_pi05_rtc.yaml``，填写 checkpoint、机器人序列号、左右臂、两路相机序列号、prompt、控制频率、单位、参考坐标系、限幅和夹爪范围。Compact 数据中的夹爪 state 与 command 都是以米表示的开口宽度，因此示例把两项 range 均设为 ``[0.003, 0.060]``。``policy.camera_keys`` 把 Flexiv 的 ``main`` 与 ``wrist`` 图像映射到训练使用的准确 key。policy、horizon、输出维度和物理语义通过检查之后，``FlexivEnv`` 才会打开硬件资源。

配置完成后运行：

.. code-block:: bash

   python examples/embodiment/run_flexiv_pi05_rtc.py \
       --config /absolute/path/to/flexiv_pi05_rtc.yaml

控制与推理时序
--------------

第一次 inference 在稳定控制开始前同步完成，并且不携带 previous-action guidance。稳定阶段中，owner thread 先接收已完成的 inference，再读取缓存的相机图像和机器人 state，随后再次接收结果；它至多提交一个不可变 inference request，再轮询一次结果，然后选择一项 physical delta，发送最终 TCP target 和独立夹爪命令，提交当前 tick，最后等待绝对 monotonic deadline。三次非阻塞轮询允许在 observation 采集期间或提交后完成的结果于当前 tick 切换，避免固定增加一个控制周期。

Inference worker 不调用机器人，也不修改 cursor。每个 request 记录 session、generation、observation tick、submit tick、提交时的 committed tick 和旧 cursor。接收结果时，``stale_prefix`` 等于提交后已经 committed 的控制 tick 数；scheduler 丢弃这些项后再安装新 chunk。下一次 RTC request 只使用上一 normalized raw chunk 尚未执行的尾部，机器人只接收 postprocessed physical chunk。

每项 physical delta 都相对于该 chunk observation 中保存的 measured TCP pose。adapter 用选中的 delta 计算最终 TCP target 并覆盖上一 target，因此 chunk 后续行不会逐项累加。hold 只重发最终 target，不会再次应用 delta。Flexiv NRT 手臂命令与 latest-command 夹爪 worker 不会等待控制循环；后台相机线程持续保存两路最新完整 RGB frame。正常控制 tick 因而不等待 policy、夹爪 ``Move``、相机 ``wait_for_frames``，也不使用相对周期 sleep。

不连接硬件的验证
----------------

测试使用带门闩的 fake policy、fake robot 和可控时钟，覆盖 RTC guidance、跨越 0/1/多个 tick 的 inference、精确 chunk 切换、单一在途 request、generation reset、hold、绝对 deadline、TCP 合成、夹爪独立控制和幂等清理。

.. code-block:: bash

   pytest -q tests/unit_tests/test_openpi_rtc.py
   pytest -q tests/unit_tests/test_real_env.py -k flexiv

这些测试不验证 GPU checkpoint、相机序列号、所选机械臂、夹爪标定或机器人工作空间限位。首次真机运行前，应在低速条件下核对这些参数。
