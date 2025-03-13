# ComfyUI Gemini Pro Node

这是一个用于 ComfyUI 的 Google Gemini Pro API 集成节点，支持文本、图像、视频和音频输入。

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

3. 安装依赖：
```bash
pip install google-generativeai pillow numpy torch
 ```
```

## 使用方法
1. 获取 Google Gemini Pro API Key：
   
   - 访问 Google AI Studio
   - 创建新的 API Key
2. 在节点中配置：
   
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
## 许可证
MIT License

创建 `LICENSE` 文件：


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