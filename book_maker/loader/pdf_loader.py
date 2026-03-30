import hashlib
import html
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
from ebooklib import epub

from book_maker.utils import prompt_config_to_kwargs

from .base_loader import BaseBookLoader
from .helper import not_trans


@dataclass
class DocumentSegment:
    segment_id: str
    page_number: int
    block_index: int
    text: str


@dataclass
class DocumentCheckpoint:
    source_path: str
    source_sha256: str
    config: dict[str, Any]
    config_fingerprint: str
    translations: dict[str, str] = field(default_factory=dict)
    completed_count: int = 0

    def to_dict(self):
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "translations": self.translations,
            "completed_count": self.completed_count,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            source_path=payload.get("source_path", ""),
            source_sha256=payload.get("source_sha256", ""),
            config=payload.get("config", {}),
            config_fingerprint=payload.get("config_fingerprint", ""),
            translations=dict(payload.get("translations", {})),
            completed_count=int(payload.get("completed_count", 0)),
        )


class PDFBookLoader(BaseBookLoader):
    document_format = "pdf"

    def __init__(
        self,
        pdf_name,
        model,
        key,
        resume,
        language,
        model_api_base=None,
        is_test=False,
        test_num=5,
        prompt_config=None,
        single_translate=False,
        context_flag=False,
        context_paragraph_limit=0,
        temperature=1.0,
        source_lang="auto",
        parallel_workers=1,
    ) -> None:
        if fitz is None:
            raise Exception("PyMuPDF (fitz) is required to use PDF loader")

        self.book_path = str(pdf_name)
        self.pdf_name = self.book_path
        self.translate_model = model(
            key,
            language,
            api_base=model_api_base,
            context_flag=context_flag,
            context_paragraph_limit=context_paragraph_limit,
            temperature=temperature,
            source_lang=source_lang,
            **prompt_config_to_kwargs(prompt_config),
        )
        self.resume = resume
        self.is_test = is_test
        self.test_num = test_num
        self.batch_size = 8
        self.single_translate = single_translate
        self.context_flag = context_flag
        self.parallel_workers = max(1, parallel_workers)
        self.selected_model_name = None
        self.source_lang = source_lang
        self.temperature = temperature
        self.translation_style = ""
        self.origin_book = self._extract_segments()
        self.checkpoint_path = (
            Path(self.book_path).parent
            / f".{Path(self.book_path).stem}.translation_state.json"
        )
        self.legacy_bin_path = (
            Path(self.book_path).parent / f".{Path(self.book_path).stem}.temp.bin"
        )
        self._source_sha256 = self._hash_source_file()
        self._checkpoint_state = self._default_checkpoint_state()

    def _make_new_book(self, book):
        pass

    def _hash_source_file(self):
        hasher = hashlib.sha256()
        with open(self.book_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _build_runtime_config(self):
        config = {
            "document_format": self.document_format,
            "language": getattr(self.translate_model, "language", None),
            "translator_class": self.translate_model.__class__.__name__,
            "selected_model_name": self.selected_model_name,
            "translator_model": getattr(self.translate_model, "model", None),
            "prompt_template": getattr(self.translate_model, "prompt_template", None),
            "prompt_sys_msg": getattr(self.translate_model, "prompt_sys_msg", None),
            "source_lang": self.source_lang,
            "temperature": self.temperature,
            "single_translate": self.single_translate,
            "batch_size": self.batch_size,
            "context_flag": self.context_flag,
        }
        fingerprint = hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return config, fingerprint

    def _default_checkpoint_state(self):
        config, fingerprint = self._build_runtime_config()
        return DocumentCheckpoint(
            source_path=os.path.abspath(self.book_path),
            source_sha256=self._source_sha256,
            config=config,
            config_fingerprint=fingerprint,
        )

    @staticmethod
    def _normalize_block_text(text):
        paragraphs = []
        current = ""
        for raw_line in text.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line:
                if current:
                    paragraphs.append(current)
                    current = ""
                continue
            if not current:
                current = line
                continue
            if current.endswith("-"):
                current = f"{current[:-1]}{line}"
            else:
                current = f"{current} {line}"
        if current:
            paragraphs.append(current)
        return "\n".join(paragraphs).strip()

    def _extract_segments(self):
        segments = []
        document = None
        try:
            document = fitz.open(self.book_path)
            for page_number, page in enumerate(document, start=1):
                raw_blocks = page.get_text("blocks") or []
                ordered_blocks = sorted(
                    raw_blocks,
                    key=lambda block: (
                        block[1] if len(block) > 1 else 0,
                        block[0] if len(block) > 0 else 0,
                    ),
                )
                page_block_index = 0
                for block in ordered_blocks:
                    if len(block) < 5:
                        continue
                    block_type = block[6] if len(block) > 6 else 0
                    if block_type not in (None, 0):
                        continue
                    text = self._normalize_block_text(str(block[4]))
                    if not text:
                        continue
                    if self._is_special_text(text) or not_trans(text):
                        continue
                    page_block_index += 1
                    segments.append(
                        DocumentSegment(
                            segment_id=f"page-{page_number}#block-{page_block_index}",
                            page_number=page_number,
                            block_index=page_block_index,
                            text=text,
                        )
                    )
        except Exception as e:
            raise Exception("can not load file") from e
        finally:
            if document is not None:
                close = getattr(document, "close", None)
                if callable(close):
                    close()
        return segments

    def _load_json_checkpoint_state(self):
        if not self.checkpoint_path.exists():
            return None

        with open(self.checkpoint_path, encoding="utf-8") as f:
            payload = json.load(f)

        checkpoint = DocumentCheckpoint.from_dict(payload)
        expected = self._default_checkpoint_state()
        if checkpoint.source_path != expected.source_path:
            raise ValueError("checkpoint source path does not match current book")
        if checkpoint.source_sha256 != expected.source_sha256:
            raise ValueError("checkpoint source file has changed")
        if checkpoint.config_fingerprint != expected.config_fingerprint:
            raise ValueError("checkpoint config does not match current parameters")
        return checkpoint

    def _load_legacy_checkpoint_state(self):
        if not self.legacy_bin_path.exists():
            return None

        checkpoint = self._default_checkpoint_state()
        with open(self.legacy_bin_path, encoding="utf-8") as f:
            legacy_lines = f.read().splitlines()

        for segment, translation in zip(self.origin_book, legacy_lines):
            checkpoint.translations[segment.segment_id] = translation
        checkpoint.completed_count = len(checkpoint.translations)
        return checkpoint

    def load_state(self):
        checkpoint = self._load_json_checkpoint_state()
        if checkpoint is None:
            checkpoint = self._load_legacy_checkpoint_state()
        if checkpoint is None:
            raise Exception("can not load resume file")
        self._checkpoint_state = checkpoint

    def _save_json_checkpoint(self):
        temp_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                self._checkpoint_state.to_dict(),
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        os.replace(temp_path, self.checkpoint_path)

    def _save_progress(self):
        self._checkpoint_state.completed_count = len(self._checkpoint_state.translations)
        self._save_json_checkpoint()

    def _build_output_pairs(self, include_partial=False):
        pairs = []
        segments = self.origin_book[: self.test_num] if self.is_test else self.origin_book
        for segment in segments:
            translation = self._checkpoint_state.translations.get(segment.segment_id)
            if not self.single_translate:
                pairs.append(segment.text)
            if translation is not None:
                pairs.append(translation)
            elif include_partial and self.single_translate:
                pairs.append("")
        return pairs

    def _segments_with_translations(self, include_partial=False):
        items = []
        segments = self.origin_book[: self.test_num] if self.is_test else self.origin_book
        for segment in segments:
            translation = self._checkpoint_state.translations.get(segment.segment_id)
            if translation is None and not include_partial:
                continue
            items.append((segment, translation))
        return items

    def save_file(self, book_path, content):
        try:
            with open(book_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(content))
        except Exception as e:
            raise Exception("can not save file") from e

    def _write_epub(self, book_path, include_partial=False):
        segments = self._segments_with_translations(include_partial=include_partial)
        if not segments:
            return False

        try:
            book = epub.EpubBook()
            title = Path(self.book_path).stem
            book.set_identifier(title)
            book.set_title(title)
            book.set_language(
                self.translate_model.language
                if hasattr(self.translate_model, "language")
                else "en"
            )

            style = """
body { font-family: serif; }
section.page { margin: 0 0 1.8em 0; }
h2.page-label { font-size: 1.05em; margin: 1.2em 0 0.5em 0; }
p.original { margin: 0.55em 0 0.2em 0; }
p.translation { margin: 0 0 0.85em 0; }
"""
            chapter = epub.EpubHtml(title=title, file_name="index.xhtml", lang="en")

            body_parts = [
                "<html><head>",
                "<meta charset='utf-8'/>",
                f"<style>{style}</style>",
                "</head><body>",
            ]
            current_page = None
            for segment, translation in segments:
                if current_page != segment.page_number:
                    if current_page is not None:
                        body_parts.append("</section>")
                    current_page = segment.page_number
                    body_parts.append(
                        f"<section class='page' data-page='{segment.page_number}'>"
                    )
                    body_parts.append(
                        f"<h2 class='page-label'>Page {segment.page_number}</h2>"
                    )
                if not self.single_translate:
                    body_parts.append(
                        "<p class='original'>"
                        f"{html.escape(segment.text).replace(chr(10), '<br/>')}"
                        "</p>"
                    )
                if translation is not None:
                    body_parts.append(
                        "<p class='translation'>"
                        f"{html.escape(translation).replace(chr(10), '<br/>')}"
                        "</p>"
                    )
            if current_page is not None:
                body_parts.append("</section>")
            body_parts.append("</body></html>")
            chapter.content = "".join(body_parts)

            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.toc = (chapter,)
            book.spine = ["nav", chapter]
            epub.write_epub(str(book_path), book)
            return True
        except Exception as e:
            print(f"create epub failed: {e}")
            return False

    def _save_temp_book(self):
        txt_path = (
            Path(self.book_path).parent / f"{Path(self.book_path).stem}_bilingual_temp.txt"
        )
        self.save_file(str(txt_path), self._build_output_pairs(include_partial=True))
        epub_path = (
            Path(self.book_path).parent
            / f"{Path(self.book_path).stem}_bilingual_temp.epub"
        )
        self._write_epub(epub_path, include_partial=True)

    def _save_final_book(self):
        txt_path = (
            Path(self.book_path).parent / f"{Path(self.book_path).stem}_bilingual.txt"
        )
        self.save_file(str(txt_path), self._build_output_pairs())

        epub_path = (
            Path(self.book_path).parent / f"{Path(self.book_path).stem}_bilingual.epub"
        )
        if self._write_epub(epub_path):
            print(f"created epub: {Path(self.book_path).stem}_bilingual.epub")
        else:
            print(
                "epub creation skipped or failed; bilingual text saved to txt fallback"
            )

    def _translate_batch(self, batch):
        texts = [segment.text for segment in batch]
        translations = self.translate_model.translate_list(texts)
        if len(translations) != len(texts):
            translations = [self.translate_model.translate(text) for text in texts]
        return translations

    def make_bilingual_book(self):
        segments = self.origin_book[: self.test_num] if self.is_test else self.origin_book
        if self.resume:
            self.load_state()
        else:
            self._checkpoint_state = self._default_checkpoint_state()

        try:
            pending = [
                segment
                for segment in segments
                if segment.segment_id not in self._checkpoint_state.translations
            ]
            for start in range(0, len(pending), self.batch_size):
                batch = pending[start : start + self.batch_size]
                if not batch:
                    continue
                translations = self._translate_batch(batch)
                for segment, translation in zip(batch, translations):
                    self._checkpoint_state.translations[segment.segment_id] = translation
                self._save_progress()

            self._save_final_book()
        except (KeyboardInterrupt, Exception) as e:
            print(e)
            print("you can resume it next time")
            self._save_progress()
            self._save_temp_book()
            sys.exit(0)


class MOBIBookLoader(PDFBookLoader):
    document_format = "mobi"
