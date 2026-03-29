from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=None)
def _load_symbol(module_name, symbol_name):
    module = import_module(module_name)
    return getattr(module, symbol_name)


class _LazyImportDict(dict):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, tuple):
            return _load_symbol(*value)
        return value

    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]

    def items(self):
        for key in super().keys():
            yield key, self[key]

    def values(self):
        for key in super().keys():
            yield self[key]


MODEL_DICT = _LazyImportDict(
    {
        "openai": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "chatgptapi": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "gpt4": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "gpt4omini": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "gpt4o": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "gpt5mini": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "o1preview": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "o1": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "o1mini": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "o3mini": ("book_maker.translator.chatgptapi_translator", "ChatGPTAPI"),
        "google": ("book_maker.translator.google_translator", "Google"),
        "caiyun": ("book_maker.translator.caiyun_translator", "Caiyun"),
        "deepl": ("book_maker.translator.deepl_translator", "DeepL"),
        "deeplfree": ("book_maker.translator.deepl_free_translator", "DeepLFree"),
        "claude": ("book_maker.translator.claude_translator", "Claude"),
        "claude-sonnet-4-6": ("book_maker.translator.claude_translator", "Claude"),
        "claude-opus-4-6": ("book_maker.translator.claude_translator", "Claude"),
        "claude-opus-4-5-20251101": ("book_maker.translator.claude_translator", "Claude"),
        "claude-haiku-4-5-20251001": ("book_maker.translator.claude_translator", "Claude"),
        "claude-sonnet-4-5-20250929": ("book_maker.translator.claude_translator", "Claude"),
        "claude-opus-4-1-20250805": ("book_maker.translator.claude_translator", "Claude"),
        "claude-opus-4-20250514": ("book_maker.translator.claude_translator", "Claude"),
        "claude-sonnet-4-20250514": ("book_maker.translator.claude_translator", "Claude"),
        "gemini": ("book_maker.translator.gemini_translator", "Gemini"),
        "geminipro": ("book_maker.translator.gemini_translator", "Gemini"),
        "groq": ("book_maker.translator.groq_translator", "GroqClient"),
        "tencentransmart": (
            "book_maker.translator.tencent_transmart_translator",
            "TencentTranSmart",
        ),
        "customapi": ("book_maker.translator.custom_api_translator", "CustomAPI"),
        "xai": ("book_maker.translator.xai_translator", "XAIClient"),
        "qwen": ("book_maker.translator.qwen_translator", "QwenTranslator"),
        "qwen-mt-turbo": (
            "book_maker.translator.qwen_translator",
            "QwenTranslator",
        ),
        "qwen-mt-plus": ("book_maker.translator.qwen_translator", "QwenTranslator"),
        # add more here
    }
)
