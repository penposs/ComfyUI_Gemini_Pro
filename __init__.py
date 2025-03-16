from .Gemini_Pro_Node import GeminiProNode, GeminiFileUpload, GeminiFileProcessing
from .Gemini_Pro_Editimage import GeminiProEditimage
from .Gemini_Pro_Chat import GeminiProChat


NODE_CLASS_MAPPINGS = {
    "Gemini Pro": GeminiProNode,
    "Gemini-Pro-Editimage": GeminiProEditimage,
    "Gemini-Pro-Chat": GeminiProChat,
    "Gemini File Upload": GeminiFileUpload,
    "Gemini File Processing": GeminiFileProcessing
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini Pro": "Gemini Pro",
    "Gemini-Pro-Editimage": "Gemini Pro 图像编辑器",
    "Gemini-Pro-Chat": "Gemini Pro 对话图文",
    "Gemini File Upload": "gemini-pro-文件上传",
    "Gemini File Processing": "gemini-pro-文件处理"
}

# 注册自定义文件类型
CUSTOM_NODE_TYPE_MAPPINGS = {
    "GEMINI_FILE": {"default": None}
} 