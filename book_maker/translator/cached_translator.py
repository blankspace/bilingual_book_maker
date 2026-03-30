import hashlib
import json
import os
from pathlib import Path


class CachedTranslator:
    def __init__(self, translator, cache_path, cache_context):
        object.__setattr__(self, "_translator", translator)
        object.__setattr__(self, "_cache_path", Path(cache_path))
        object.__setattr__(self, "_cache_context", dict(cache_context))
        object.__setattr__(self, "_dirty", False)
        object.__setattr__(self, "_cache", self._load_cache())

    @classmethod
    def from_book_path(cls, translator, book_path, cache_context):
        path = Path(book_path)
        cache_path = path.parent / f".{path.stem}.translation_cache.json"
        return cls(translator, cache_path, cache_context)

    def __getattr__(self, name):
        return getattr(self._translator, name)

    def __setattr__(self, name, value):
        if name in {"_translator", "_cache_path", "_cache_context", "_cache", "_dirty"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._translator, name, value)

    def _load_cache(self):
        if not self._cache_path.exists():
            return {}
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        if payload.get("context") != self._cache_context:
            return {}
        translations = payload.get("translations", {})
        return translations if isinstance(translations, dict) else {}

    def _save_cache(self):
        if not self._dirty:
            return
        payload = {
            "version": 1,
            "context": self._cache_context,
            "translations": self._cache,
        }
        temp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temp_path, self._cache_path)
        object.__setattr__(self, "_dirty", False)

    def _cache_enabled(self):
        return not self._cache_context.get("context_flag", False)

    def _cache_key(self, text):
        payload = json.dumps(
            {"context": self._cache_context, "text": text},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached_translation(self, text):
        if not self._cache_enabled():
            return None
        return self._cache.get(self._cache_key(text))

    def _store_translation(self, text, translation):
        if not self._cache_enabled():
            return translation
        key = self._cache_key(text)
        self._cache[key] = translation
        object.__setattr__(self, "_dirty", True)
        return translation

    def translate(self, text, *args, **kwargs):
        cached_translation = self._get_cached_translation(text)
        if cached_translation is not None:
            return cached_translation

        translation = self._translator.translate(text, *args, **kwargs)
        translation = self._store_translation(text, translation)
        self._save_cache()
        return translation

    def translate_list(self, text_list):
        if not self._cache_enabled():
            return self._translator.translate_list(text_list)

        results = [None] * len(text_list)
        missing_indices = []
        missing_texts = []

        for index, text in enumerate(text_list):
            cached_translation = self._get_cached_translation(text)
            if cached_translation is not None:
                results[index] = cached_translation
            else:
                missing_indices.append(index)
                missing_texts.append(text)

        if missing_texts:
            translated_texts = self._translator.translate_list(missing_texts)
            if len(translated_texts) != len(missing_texts):
                translated_texts = [
                    self._translator.translate(text) for text in missing_texts
                ]

            for index, text, translation in zip(
                missing_indices, missing_texts, translated_texts
            ):
                results[index] = self._store_translation(text, translation)
            self._save_cache()

        return results

    def translate_segments(self, segments):
        if not self._cache_enabled():
            return self._translator.translate_segments(segments)

        results = {}
        missing_segments = []
        for segment in segments:
            cached_translation = self._get_cached_translation(segment["text"])
            if cached_translation is None:
                missing_segments.append(segment)
            else:
                results[segment["id"]] = cached_translation

        if missing_segments:
            translated_segments = self._translator.translate_segments(missing_segments)
            for translated_segment, source_segment in zip(
                translated_segments, missing_segments
            ):
                translation = translated_segment.get("translation")
                if translation is None:
                    continue
                results[source_segment["id"]] = self._store_translation(
                    source_segment["text"], translation
                )
            self._save_cache()

        return [
            {"id": segment["id"], "translation": results[segment["id"]]}
            for segment in segments
            if segment["id"] in results
        ]
