from .Gemini_Pro_Node import GeminiProNode, GeminiFileUpload, GeminiFileProcessing


NODE_CLASS_MAPPINGS = {
    "Gemini Pro": GeminiProNode,
    "Gemini File Upload": GeminiFileUpload,
    "Gemini File Processing": GeminiFileProcessing
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini Pro": "Gemini Pro",
    "Gemini File Upload": "gemini-pro-文件上传",
    "Gemini File Processing": "gemini-pro-文件处理"
}

# 注册自定义文件类型
CUSTOM_NODE_TYPE_MAPPINGS = {
    "GEMINI_FILE": {"default": None}
}