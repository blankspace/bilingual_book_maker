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


BOOK_LOADER_DICT = _LazyImportDict(
    {
        "epub": ("book_maker.loader.epub_loader", "EPUBBookLoader"),
        "txt": ("book_maker.loader.txt_loader", "TXTBookLoader"),
        "srt": ("book_maker.loader.srt_loader", "SRTBookLoader"),
        "md": ("book_maker.loader.md_loader", "MarkdownBookLoader"),
        "pdf": ("book_maker.loader.pdf_loader", "PDFBookLoader"),
        # TODO add more here
    }
)
