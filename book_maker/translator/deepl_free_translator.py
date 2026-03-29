import time
import random
import re

import httpx

from book_maker.utils import LANGUAGES, TO_LANGUAGE_CODE, safe_rich_print

from .base_translator import Base
from .google_translator import Google
from PyDeepLX import PyDeepLX

print = safe_rich_print


class DeepLFree(Base):
    """
    DeepL free translator
    """

    def __init__(self, key, language, **kwargs) -> None:
        super().__init__(key, language)
        l = language if language in LANGUAGES else TO_LANGUAGE_CODE.get(language)
        if l not in [
            "bg",
            "zh",
            "cs",
            "da",
            "nl",
            "en-US",
            "en-GB",
            "et",
            "fi",
            "fr",
            "de",
            "el",
            "hu",
            "id",
            "it",
            "ja",
            "lv",
            "lt",
            "pl",
            "pt-PT",
            "pt-BR",
            "ro",
            "ru",
            "sk",
            "sl",
            "es",
            "sv",
            "tr",
            "uk",
            "ko",
            "nb",
        ]:
            raise Exception(f"DeepL do not support {l}")
        self.language = l
        self.time_random = [0.3, 0.5, 1, 1.3, 1.5, 2]
        self._fallback_translator = None

    def rotate_key(self):
        pass

    def _get_fallback_translator(self):
        if self._fallback_translator is None:
            self._fallback_translator = Google("", self.language)
        return self._fallback_translator

    @staticmethod
    def _translate_with_httpx_compat(text, source_lang, target_lang):
        try:
            return str(PyDeepLX.translate(text, source_lang, target_lang))
        except TypeError as exc:
            if "unexpected keyword argument 'proxies'" not in str(exc):
                raise

            original_client = PyDeepLX.httpx.Client

            def compat_client(*args, **kwargs):
                proxies = kwargs.pop("proxies", None)
                if proxies is not None:
                    kwargs.setdefault("proxy", proxies)
                return original_client(*args, **kwargs)

            try:
                PyDeepLX.httpx.Client = compat_client
                return str(PyDeepLX.translate(text, source_lang, target_lang))
            finally:
                PyDeepLX.httpx.Client = original_client

    def translate(self, text):
        print(text)
        try:
            t_text = self._translate_with_httpx_compat(text, "EN", self.language)
        except Exception as exc:
            print(
                f"[yellow]DeepL Free translation failed ({type(exc).__name__}: {exc}). Falling back to Google Translate.[/yellow]"
            )
            t_text = self._get_fallback_translator().translate(text)
        # spider rule
        time.sleep(random.choice(self.time_random))
        print("[bold green]" + re.sub("\n{3,}", "\n\n", t_text) + "[/bold green]")
        return t_text
