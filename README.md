# ComfyUI Gemini Pro Node

这是一个用于 ComfyUI 的 Google Gemini Pro API 全功能集合节点支持文本、图像、视频和音频输入。

支持最新的图像生成和编辑能力\剧本生成
![dee31ff12e6960cb7b7f6f262cc71ca](https://github.com/user-attachments/assets/d242dd82-f525-410a-8b9a-ebc76e41ba71)

![6e379a92710d818f4a9d3be9d6dab17](https://github.com/user-attachments/assets/88195a23-2235-4356-9548-eefea76f779c)

![6f43e715a7814333f021d5839e35420](https://github.com/user-attachments/assets/65be0222-1357-4f8d-8b40-644bb2952589)
![bbcebd0c20b384578da722b3b28f1ee](https://github.com/user-attachments/assets/5ceb700d-871e-498e-ab27-ddccb480bcb4)



## 功能特点

- 支持多种输入类型：文本、图像、视频、音频
- 支持系统提示词配置
- 内置重试机制和超时控制
- 支持代理设置
- 支持自定义温度和最大输出令牌数
- 配置文件持久化

## 安装

1. 确保已安装 ComfyUI
2. 克隆本仓库到 ComfyUI 的 custom_nodes 目录：
```bash
cd custom_nodes
git clone https://github.com/penposs/ComfyUI_Gemini_Pro.git
```

3. 安装依赖：
```bash
# 先卸载现有的可能冲突的库
pip uninstall -y google-api-python-client google-generativeai 

# 安装最新版本
pip install google-generativeai --upgrade
```

## 使用方法
1. 获取 Google Gemini Pro API Key：
2. https://aistudio.google.com/
   
   - 访问 Google AI Studio
   - 创建新的 API Key
3. 在节点中配置：
   
   - 输入你的 API Key
   - 选择输入类型（文本/图像/视频/音频）
   - 设置系统提示词和用户提示词
   - 根据需要调整其他参数
## 参数说明
- prompt : 用户提示词
- system_prompt : 系统提示词
- input_type : 输入类型（text/image/video/audio）
- model : Gemini 模型版本
- api_key : Google Gemini Pro API Key
- proxy : 代理服务器地址（可选）
- delay_time : 请求延迟时间
- max_output_tokens : 最大输出令牌数
- temperature : 生成温度（0.0-1.0）
## 注意事项
- 请妥善保管你的 API Key
- 建议使用代理时进行测试
- 图像和视频输入会自动进行大小调整
- API 调用有 30 秒超时限制

## 联系方式
- B站：[@penposs](https://space.bilibili.com/3493282531248138?spm_id_from=333.788.0.0)

---

# ComfyUI Gemini Pro Node

This is a comprehensive Google Gemini Pro API node collection for ComfyUI that supports text, image, video, and audio inputs.

Supports the latest image generation and editing capabilities.

## Features

- Supports multiple input types: text, image, video, audio
- System prompt configuration
- Built-in retry mechanism and timeout control
- Proxy settings support
- Custom temperature and maximum output token settings
- Configuration file persistence

## Installation

1. Make sure ComfyUI is installed
2. Clone this repository to the custom_nodes directory of ComfyUI:
```bash
cd custom_nodes
git clone https://github.com/penposs/ComfyUI_Gemini_Pro.git
```

3. Install dependencies:
```bash
pip install google-generativeai pillow numpy torch
```

## Usage
1. Get a Google Gemini Pro API Key:
   
   - Visit Google AI Studio
   - Create a new API Key
2. Configure in the node:
   
   - Enter your API Key
   - Select input type (text/image/video/audio)
   - Set system prompt and user prompt
   - Adjust other parameters as needed
## Parameter Description
- prompt: User prompt
- system_prompt: System prompt
- input_type: Input type (text/image/video/audio)
- model: Gemini model version
- api_key: Google Gemini Pro API Key
- proxy: Proxy server address (optional)
- delay_time: Request delay time
- max_output_tokens: Maximum output tokens
- temperature: Generation temperature (0.0-1.0)
## Notes
- Please keep your API Key secure
- Testing with a proxy is recommended
- Image and video inputs will be automatically resized
- API calls have a 30-second timeout limit

## Contact
- Bilibili: [@penposs](https://space.bilibili.com/3493282531248138?spm_id_from=333.788.0.0)

## License

MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


感谢 / Thanks to:
- ZHO: https://github.com/ZHO-ZHO-ZHO/ComfyUI-Gemini
- CY-CHENYUE: https://github.com/CY-CHENYUE/ComfyUI-Gemini-API
