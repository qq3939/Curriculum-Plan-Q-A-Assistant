# 培养计划学业问答助手

本项目是一个面向学生自查的本地网页助手，资料源固定为 `培养计划` 文件夹中的 2025 级培养计划 PDF。

## 功能

- 根据培养计划回答专业培养目标、毕业要求、课程、学分、建议修读学期等问题。
- 对学业相关问题给出建议，例如大类内专业选择、学期选课重点、学习路径安排。
- 回答会展示来源文件、页码、章节和原文片段。
- 学业建议会区分“资料依据”和“基于学生情况的建议”。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 模型配置

复制 `.env.example` 为 `.env`，填写 OpenAI-compatible API 配置：

```powershell
Copy-Item .env.example .env
```

必要变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`

如果使用 DeepSeek 聊天模型，可按下面配置：

```powershell
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_CHAT_MODEL=deepseek-v4-flash
OPENAI_EMBEDDING_MODEL=local-hash
```

`OPENAI_EMBEDDING_MODEL=local-hash` 表示检索索引用本地向量，聊天回答仍调用云端 API。未配置 API 时，应用会使用本地简易向量检索显示相关原文片段，但不会生成完整自然语言答案。

## 构建索引

```powershell
python scripts\build_index.py
```

索引会写入 `data/index`。如果培养计划 PDF 或 embedding 配置发生变化，应用会自动重建索引。

## 运行网页

```powershell
streamlit run app.py
```

## API 连通性检查

```powershell
python scripts\check_api.py
```

该脚本会调用 embedding 和 chat 接口各一次；不配置 `OPENAI_API_KEY` 时会明确提示。

## 样例检查

```powershell
python scripts\eval_samples.py
```

该脚本会检查 6 个 PDF 是否被读取、页数是否匹配、索引块是否包含必要元数据，并跑一组样例检索。

## 自动 Git 版本管理

```powershell
python scripts\auto_version.py --message "实现培养计划学业问答助手 MVP"
```

脚本会自动完成：

- 当前目录不是 Git 仓库时执行初始化。
- 自动创建或递增 `VERSION`。
- 更新 `CHANGELOG.md`。
- 暂存项目变更并提交。
- 为版本创建 `vX.Y.Z` 标签。

常用选项：

```powershell
python scripts\auto_version.py --bump minor --message "新增学业建议能力"
python scripts\auto_version.py --bump none --message "只提交文档更新" --no-tag
python scripts\auto_version.py --dry-run
```

接入 GitHub 远程仓库：

```powershell
python scripts\auto_version.py --message "实现 MVP" --remote-url https://github.com/USER/REPO.git --push
```

也可以在 `.env` 或系统环境变量中设置：

```powershell
$env:GITHUB_REMOTE_URL="git@github.com:USER/REPO.git"
python scripts\auto_version.py --message "实现 MVP" --push
```

`--push` 会推送当前分支并同步 tags。推送前请确保本机 GitHub HTTPS 或 SSH 凭据可用。
