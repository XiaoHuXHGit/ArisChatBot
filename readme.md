# AIRS CHATBOT LOACL
这个是AIRS CHATBOT的本地版本，基于Python + PySide6 + moderngl（或 PyOpenGL） + 自定义渲染管线</br>
## 爱丽丝的使用方法！

你好呀老师！<br>
让爱丽丝来教你怎么使用这个软件吧！

### 1. 安装环境

首先，你需要安装爱丽丝的运行环境。<br>
爱丽丝支持的python环境的版本目前基本都支持<br>
不过爱丽丝建议使用3.10~3.13版本的python。

爱丽丝教你如何配置运行环境：<br>
1. 下载python安装包，安装到任意目录，然后拿出那个python复制到和aris_chatbot的同级文件夹。
2. 打开powershell，进入到爱丽丝的aris_chatbot目录。
3. 输入命令：`..\pythonenv\python.exe -m pip install -r requirements.txt`
4. 等待安装完成。
### 2. 运行爱丽丝

爱丽丝的运行方式很简单，打开你的napcat，双击运行你的napcat.quick.bat<br>
然后双击运行爱丽丝的"android_startup.bat"文件<br>
新建并配置好你的napcat的websocket客户端就好，别忘了端口号要改成5090哟~

### 核心思想
- 主进程：Python（完全控制）
- GUI 框架：PySide6（用于创建透明无边框窗口 + 事件循环）
- 渲染后端：moderngl（基于 OpenGL 3.3+，高性能、简洁）或 PyOpenGL
- Live2D 支持：通过 ctypes 调用 live2d-cubism-core.dll
- 3D 模型支持：加载 glTF / VRM / OBJ / FBX（通过 pygltflib、trimesh、assimp 等）
- AI 集成：直接在 Python 中运行本地大模型（如 llama-cpp-python）
- 桌面集成：透明窗口 + 鼠标穿透（PySide6 原生支持）

## AI设计
- 基于 Python 的本地 AI 集成，使用 llama-cpp-python 和 transformers 作为 AI 引擎
- AI模块集成：TTS, NLP, ASR, CV
- TTS 模块：使用 aris voice / GPT-SoVITS 作为本地 TTS 引擎
- NLP 模块：使用 llama-cpp-python 和 transformers 作为本地 NLP 引擎
- ASR 模块：使用 FunASR 作为本地 ASR 引擎
- CV 模块：使用 qwen3-vl 作为本地 CV 引擎

### AI模块细化设计
- NLP 模块：
    - 对话：使用 transformers / llama-cpp-python 作为本地 NLP 引擎， 实现对话功能，模型使用：qwen3-instruct-32B-gguf
    - 情感分析：引擎同上，模型使用：qwen3-1B-gguf，进行快速分析和情感判断，对TTS进行情感控制
    - 长时记忆存储：使用MemU进行长时向量化记忆存储，实现对话记忆功能
- TTS 模块：
    - 语音合成：使用 aris voice / GPT-SoVITS 作为本地 TTS 引擎，实现语音合成功能。


