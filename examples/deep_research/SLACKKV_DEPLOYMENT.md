# SlackKV Deep Research 环境

本 fork 的 SlackKV 基线使用 `experiment/branch-time-prediction` 分支。完整迁移步骤在同级 `slackkv-control/deployment/MIGRATION.md`，以其中固定的提交版本为准。

从数据盘设置 `UV_CACHE_DIR` 后执行 `bash examples/deep_research/install_slackkv.sh`。使用 Python 3.12 和已有 `uv.lock`，不要重新生成 lock 或安装 PyPI 上的 deepagents 来替代本地 `libs/deepagents`。安装后 `deepagents.__file__` 应指向此仓库的源码。

离线实验入口是 control 仓库的 `experiments/subagent_action_prediction/run_one.py`，不是这里的默认 `agent.py`/`run_local.py`。离线入口不要求 Tavily key，不复制旧 `.env`；设置 `LANGSMITH_TRACING=false`。

当前已验证 Qwen3-8B：主 Agent 保留 think_tool，子 Agent 去掉 think_tool，在搜索后的模型可见正文输出行动及剩余搜索批次区间。保留历史预测，不用预测控制停止。gpt-oss 的 Harmony、reasoning 和预测提取尚待新服务器真实验收；不能直接使用 Qwen3 的模型参数。
