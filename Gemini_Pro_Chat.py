import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import tempfile
from io import BytesIO
import google.generativeai as genai
import time
import traceback
from typing import List, Tuple, Dict, Any, Optional
import re
from .Gemini_Pro_Editimage import GeminiProEditimage

class GeminiProChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp","models/gemini-2.5-flash-preview-05-20"], {"default": "models/gemini-2.0-flash-exp"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "response_text")
    FUNCTION = "generate_chat_response"
    CATEGORY = "Gemini-Pro"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.image_generator = GeminiProEditimage()  # 使用GeminiProEditimage进行图像生成
        
        # 自定义临时目录设置
        self.temp_dir = os.path.join(tempfile.gettempdir(), "comfyui_gemini")
        self.debug_dir = os.path.join(self.temp_dir, "gemini_debug_images")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # 清理过期缓存文件
        self.clean_old_cache()
        
    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥，其次从config.json读取"""
        # 如果用户输入了有效的密钥，使用该密钥
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 保存到config.json文件中
            try:
                config_data = {}
                if os.path.exists(self.config_file):
                    with open(self.config_file, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                
                config_data["api_key"] = user_input_key
                
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=4)
                    
                self.log("已保存API密钥到config.json")
            except Exception as e:
                self.log(f"保存API密钥到config.json失败: {e}")
            return user_input_key
            
        # 如果用户没有输入，尝试从config.json文件读取
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    
                if "api_key" in config_data and config_data["api_key"] and len(config_data["api_key"]) > 10:
                    self.log("使用config.json中的API密钥")
                    return config_data["api_key"]
            except Exception as e:
                self.log(f"读取config.json中的API密钥失败: {e}")
                
        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥，请在节点中输入API密钥或在config.json中配置")
        return ""
    
    def generate_chat_response(self, prompt, api_key, model, temperature, width, height, reference_image=None, seed=0):
        """生成对话式回应，包含文本和图像"""
        self.log_messages = []
        
        self.log(f"开始处理提示词: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                # 创建空白图像
                blank_img = Image.new('RGB', (512, 512), (255, 255, 255))
                blank_tensor = torch.from_numpy(np.array(blank_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                blank_tensor = blank_tensor.permute(0, 2, 3, 1)  # [B, H, W, C]
                return (blank_tensor, error_message)
            
            # 初始化 Gemini API
            genai.configure(api_key=actual_api_key)
            # 创建生成模型
            model_obj = genai.GenerativeModel(model)
            
            # 准备文本内容
            contents = []
            
            # 添加特殊指令，让模型返回更多图像
            enhanced_prompt = prompt
            if not ("图片" in prompt or "图像" in prompt or "配图" in prompt or "image" in prompt.lower()):
                enhanced_prompt += "\n请在回复中提供相关图片或图像示例。"
                self.log("已添加图像提示到请求")
            
            # 处理参考图像
            if reference_image is not None:
                try:
                    self.log("处理参考图像")
                    # 确保图像格式正确
                    if len(reference_image.shape) == 4:  # [B, H, W, 3] 格式
                        # 处理第一帧图像
                        input_image = reference_image[0].cpu().numpy()
                        
                        # 转换为PIL图像
                        input_image = (input_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(input_image)
                        
                        # 保存为临时文件
                        temp_img_path = os.path.join(self.temp_dir, f"reference_{int(time.time())}.png")
                        pil_image.save(temp_img_path)
                        
                        self.log(f"参考图像处理成功，尺寸: {pil_image.width}x{pil_image.height}")
                        
                        # 添加图像部分
                        with open(temp_img_path, "rb") as f:
                            image_bytes = f.read()
                        
                        contents = [
                            {"role": "user", "parts": [
                                {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
                                {"text": enhanced_prompt}
                            ]}
                        ]
                    else:
                        self.log(f"参考图像格式不正确: {reference_image.shape}")
                        contents = [{"role": "user", "parts": [{"text": enhanced_prompt}]}]
                except Exception as img_error:
                    self.log(f"参考图像处理错误: {str(img_error)}")
                    contents = [{"role": "user", "parts": [{"text": enhanced_prompt}]}]
            else:
                contents = [{"role": "user", "parts": [{"text": enhanced_prompt}]}]
            
            # 设置参数
            generation_config = {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 32,
            }
            
            # 调用API获取响应
            self.log(f"正在调用Gemini API，使用模型: {model}, 温度: {temperature}")
            
            try:
                response = model_obj.generate_content(
                    contents,
                    generation_config=generation_config,
                    stream=False
                )
                
                # 获取文本响应
                full_text = ""
                image_tensors = []
                
                if hasattr(response, 'text'):
                    full_text = response.text
                    self.log(f"从response.text获取文本，长度: {len(full_text)} 字符")
                elif hasattr(response, 'candidates') and response.candidates:
                    if hasattr(response.candidates[0].content, 'text'):
                        full_text = response.candidates[0].content.text
                        self.log(f"从candidates[0].content.text获取文本，长度: {len(full_text)} 字符")
                    elif hasattr(response.candidates[0].content, 'parts'):
                        parts_text = []
                        
                        # 处理所有部分
                        for part in response.candidates[0].content.parts:
                            # 处理文本部分
                            if hasattr(part, 'text') and part.text:
                                parts_text.append(part.text)
                                self.log(f"添加文本部分，长度: {len(part.text)} 字符")
                            
                            # 处理图像部分
                            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                                try:
                                    # 获取图像数据
                                    image_data = part.inline_data.data
                                    mime_type = getattr(part.inline_data, 'mime_type', 'image/unknown')
                                    
                                    self.log(f"找到图像数据，mime_type: {mime_type}")
                                    
                                    # 尝试打开图像
                                    pil_image = Image.open(BytesIO(image_data))
                                    self.log(f"成功打开图像, 原始尺寸: {pil_image.width}x{pil_image.height}, 模式: {pil_image.mode}")
                                    
                                    # 调整为RGB模式
                                    if pil_image.mode != 'RGB':
                                        pil_image = pil_image.convert('RGB')
                                    
                                    # 调整图像大小
                                    pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                                    
                                    # 转换为张量
                                    img_array = np.array(pil_image).astype(np.float32) / 255.0
                                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W, C]
                                    
                                    # 保存调试图像
                                    debug_img_path = os.path.join(self.debug_dir, f"api_img_{len(image_tensors)+1}_{int(time.time())}.png")
                                    pil_image.save(debug_img_path)
                                    
                                    # 添加到图像列表
                                    image_tensors.append(img_tensor)
                                    self.log(f"成功处理图像 #{len(image_tensors)}")
                                    
                                except Exception as e:
                                    self.log(f"处理图像数据时出错: {str(e)}")
                        
                        # 合并所有文本部分
                        full_text = "\n".join(parts_text)
                        self.log(f"合并后的文本长度: {len(full_text)} 字符")
                        
                        # 确保文本不为空
                        if not full_text.strip():
                            self.log("警告: 合并后的文本为空，尝试直接获取原始响应")
                            # 尝试获取原始响应文本
                            try:
                                raw_text = str(response)
                                if len(raw_text) > 0:
                                    full_text = f"API返回内容 (原始格式):\n{raw_text[:2000]}"
                                    if len(raw_text) > 2000:
                                        full_text += "...(内容过长已截断)"
                                    self.log(f"使用原始响应文本，长度: {len(full_text)} 字符")
                            except Exception as e:
                                self.log(f"获取原始响应失败: {str(e)}")
                
                # 确保返回的文本不为空
                if not full_text or not full_text.strip():
                    full_text = "API未返回文本内容或文本内容为空"
                    self.log("警告: 返回空文本")
                
                # 检查是否有图像
                if image_tensors:
                    self.log(f"成功处理 {len(image_tensors)} 张图像")
                    
                    # 合并图像为一个批次
                    if len(image_tensors) == 1:
                        combined_images = image_tensors[0]
                    else:
                        try:
                            # 合并多张图像
                            combined_images = torch.cat(image_tensors, dim=0)
                            self.log(f"合并后的图像形状: {combined_images.shape}")
                        except Exception as e:
                            self.log(f"合并图像失败: {str(e)}")
                            # 失败时返回第一张图像
                            combined_images = image_tensors[0]
                    
                    return (combined_images, full_text)
                else:
                    self.log("API响应中未找到图像，尝试使用文本生成图像")
                    
                    # 使用文本内容生成图像
                    img_tensor, img_log = self.image_generator.generate_image(
                        prompt=prompt,
                        api_key=actual_api_key,
                        model=model,
                        temperature=temperature,
                        seed=seed if seed != 0 else None,
                        image=reference_image
                    )
                    
                    return (img_tensor, full_text)
                
            except Exception as api_error:
                error_message = f"Gemini API调用失败: {str(api_error)}"
                self.log(error_message)
                
                # 创建空白图像
                blank_img = Image.new('RGB', (width, height), (255, 255, 255))
                blank_tensor = torch.from_numpy(np.array(blank_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                blank_tensor = blank_tensor.permute(0, 2, 3, 1)  # [B, H, W, C]
                
                return (blank_tensor, error_message)
            
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"生成对话响应时出错: {str(e)}")
            traceback.print_exc()
            
            # 创建空白图像
            blank_img = Image.new('RGB', (width, height), (255, 255, 255))
            blank_tensor = torch.from_numpy(np.array(blank_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            blank_tensor = blank_tensor.permute(0, 2, 3, 1)  # [B, H, W, C]
            
            return (blank_tensor, error_message)
    
    def clean_old_cache(self, max_age_hours=24):
        """清理指定时间前的缓存文件"""
        try:
            self.log(f"检查并清理超过 {max_age_hours} 小时的缓存文件")
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            files_removed = 0
            
            if os.path.exists(self.debug_dir):
                for filename in os.listdir(self.debug_dir):
                    file_path = os.path.join(self.debug_dir, filename)
                    # 只处理文件，不处理目录
                    if os.path.isfile(file_path):
                        # 获取文件的修改时间
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > max_age_seconds:
                            try:
                                os.remove(file_path)
                                files_removed += 1
                            except Exception as e:
                                self.log(f"无法删除文件 {filename}: {e}")
            
            self.log(f"缓存清理完成，共删除 {files_removed} 个过期文件")
        except Exception as e:
            self.log(f"清理缓存时出错: {e}")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Gemini-Pro-Chat": GeminiProChat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini-Pro-Chat": "Gemini Pro 对话图文"
} 
