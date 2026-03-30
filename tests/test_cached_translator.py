from book_maker.translator.cached_translator import CachedTranslator


class CountingTranslator:
    translate_calls = 0

    def __init__(self):
        self.language = "zh-hans"
        self.model = "counting-model"
        self.prompt_template = "Translate {text} to {language}"
        self.prompt_sys_msg = ""

    @classmethod
    def reset(cls):
        cls.translate_calls = 0

    def translate(self, text, *args, **kwargs):
        type(self).translate_calls += 1
        return f"ZH::{text}"

    def translate_list(self, texts):
        return [self.translate(text) for text in texts]

    def translate_segments(self, segments):
        return [
            {"id": segment["id"], "translation": self.translate(segment["text"])}
            for segment in segments
        ]


def test_cached_translator_reuses_saved_translations(tmp_path):
    cache_path = tmp_path / ".book.translation_cache.json"
    cache_context = {
        "language": "zh-hans",
        "translator_class": "CountingTranslator",
        "selected_model_name": "openai",
        "translator_model": "counting-model",
        "prompt_template": "Translate {text} to {language}",
        "prompt_sys_msg": "",
        "context_flag": False,
    }
    CountingTranslator.reset()

    first = CachedTranslator(CountingTranslator(), cache_path, cache_context)
    assert first.translate("Hello") == "ZH::Hello"
    assert CountingTranslator.translate_calls == 1

    second = CachedTranslator(CountingTranslator(), cache_path, cache_context)
    assert second.translate("Hello") == "ZH::Hello"
    assert CountingTranslator.translate_calls == 1


def test_cached_translator_invalidates_context_mismatch(tmp_path):
    cache_path = tmp_path / ".book.translation_cache.json"
    base_context = {
        "language": "zh-hans",
        "translator_class": "CountingTranslator",
        "selected_model_name": "openai",
        "translator_model": "counting-model",
        "prompt_template": "Translate {text} to {language}",
        "prompt_sys_msg": "",
        "context_flag": False,
    }
    CountingTranslator.reset()

    first = CachedTranslator(CountingTranslator(), cache_path, base_context)
    assert first.translate("Hello") == "ZH::Hello"
    assert CountingTranslator.translate_calls == 1

    second = CachedTranslator(
        CountingTranslator(),
        cache_path,
        {**base_context, "translator_model": "different-model"},
    )
    assert second.translate("Hello") == "ZH::Hello"
    assert CountingTranslator.translate_calls == 2
