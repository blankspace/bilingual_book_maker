import json
from pathlib import Path

from bs4 import BeautifulSoup as bs
from ebooklib import ITEM_DOCUMENT, epub

from book_maker.loader.epub_loader import EPUBBookLoader


def _create_epub(epub_path, paragraphs):
    book = epub.EpubBook()
    book.set_identifier("checkpoint-test")
    book.set_title("Checkpoint Test")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Chapter", file_name="chapter.xhtml", lang="en")
    chapter.content = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    book.toc = (chapter,)
    epub.write_epub(str(epub_path), book)


def _create_epub_with_html(epub_path, chapter_html):
    book = epub.EpubBook()
    book.set_identifier("checkpoint-test")
    book.set_title("Checkpoint Test")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Chapter", file_name="chapter.xhtml", lang="en")
    chapter.content = chapter_html
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    book.toc = (chapter,)
    epub.write_epub(str(epub_path), book)


def _read_paragraphs(epub_path):
    book = epub.read_epub(str(epub_path))
    document = next(book.get_items_of_type(ITEM_DOCUMENT))
    soup = bs(document.content, "html.parser")
    return [paragraph.get_text() for paragraph in soup.find_all("p")]


def _read_tag_texts(epub_path, tag_names):
    book = epub.read_epub(str(epub_path))
    document = next(book.get_items_of_type(ITEM_DOCUMENT))
    soup = bs(document.content, "html.parser")
    return [node.get_text() for node in soup.find_all(tag_names)]


class BatchEchoModel:
    batch_calls = 0
    single_calls = 0

    def __init__(
        self,
        key,
        language,
        api_base=None,
        context_flag=False,
        context_paragraph_limit=0,
        temperature=1.0,
        source_lang="auto",
        **kwargs,
    ):
        self.language = language
        self.context_flag = context_flag
        self.prompt_template = "Translate {text} to {language}"
        self.prompt_sys_msg = ""

    @classmethod
    def reset(cls):
        cls.batch_calls = 0
        cls.single_calls = 0

    def rotate_key(self):
        pass

    def translate(self, text):
        type(self).single_calls += 1
        return f"ZH::{text}"

    def translate_segments(self, segments):
        type(self).batch_calls += 1
        return [
            {"id": segment["id"], "translation": f"ZH::{segment['text']}"}
            for segment in segments
        ]


class InvalidBatchModel(BatchEchoModel):
    @classmethod
    def reset(cls):
        super().reset()

    def translate_segments(self, segments):
        type(self).batch_calls += 1
        return [{"id": "wrong-id", "translation": "bad"}]


class ResumableModel(BatchEchoModel):
    interrupt_after = None
    translate_calls = 0

    @classmethod
    def reset(cls, interrupt_after=None):
        super().reset()
        cls.interrupt_after = interrupt_after
        cls.translate_calls = 0

    def translate(self, text):
        type(self).translate_calls += 1
        if (
            type(self).interrupt_after is not None
            and type(self).translate_calls > type(self).interrupt_after
        ):
            raise KeyboardInterrupt()
        return super().translate(text)

    def translate_segments(self, segments):
        return [
            {"id": segment["id"], "translation": self.translate(segment["text"])}
            for segment in segments
        ]


def test_epub_segment_ids_are_stable_and_checkpoint_roundtrip_preserves_newlines(
    tmp_path,
):
    epub_path = tmp_path / "stable.epub"
    _create_epub(epub_path, ["First line", "Second line"])

    loader = EPUBBookLoader(
        str(epub_path),
        BatchEchoModel,
        key="",
        resume=False,
        language="zh-hans",
    )
    loader.exclude_filelist = "nav.xhtml"

    first_ids = [
        segment.segment_id
        for context in loader._collect_document_contexts()
        for segment in context["segments"]
    ]
    second_ids = [
        segment.segment_id
        for context in loader._collect_document_contexts()
        for segment in context["segments"]
    ]

    assert first_ids == second_ids == ["chapter.xhtml#1", "chapter.xhtml#2"]

    loader._checkpoint_state = loader._default_checkpoint_state()
    loader._checkpoint_state.translations[first_ids[0]] = "line 1\nline 2"
    loader._checkpoint_state.completed_count = 1
    loader._save_json_checkpoint()

    loader.resume = True
    reloaded = loader._load_json_checkpoint_state()
    assert reloaded.translations[first_ids[0]] == "line 1\nline 2"


def test_epub_batch_translation_preserves_pairs_and_disables_parallel(tmp_path):
    epub_path = tmp_path / "parallel.epub"
    _create_epub(epub_path, ["One", "Two", "Three"])
    BatchEchoModel.reset()

    loader = EPUBBookLoader(
        str(epub_path),
        BatchEchoModel,
        key="",
        resume=False,
        language="zh-hans",
        parallel_workers=4,
    )
    loader.exclude_filelist = "nav.xhtml"
    loader.accumulated_num = 400
    loader.make_bilingual_book()

    output_path = tmp_path / "parallel_bilingual.epub"
    assert output_path.exists()
    assert loader.parallel_workers == 1
    assert loader.enable_parallel is False
    assert BatchEchoModel.batch_calls >= 1
    assert _read_paragraphs(output_path) == [
        "One",
        "ZH::One",
        "Two",
        "ZH::Two",
        "Three",
        "ZH::Three",
    ]


def test_epub_default_translate_tags_cover_headings_and_list_items(tmp_path):
    epub_path = tmp_path / "structure.epub"
    _create_epub_with_html(
        epub_path,
        "<h1>The Intelligent Owner</h1>"
        "<ol><li>Is the management reasonably efficient?</li>"
        "<li>Are the interests of the average outside shareholder receiving proper recognition?</li></ol>"
        "<p>Body paragraph.</p>",
    )
    BatchEchoModel.reset()

    loader = EPUBBookLoader(
        str(epub_path),
        BatchEchoModel,
        key="",
        resume=False,
        language="zh-hans",
    )
    loader.exclude_filelist = "nav.xhtml"
    loader.accumulated_num = 400
    loader.make_bilingual_book()

    output_path = tmp_path / "structure_bilingual.epub"
    assert output_path.exists()
    assert _read_tag_texts(output_path, ["h1", "li", "p"]) == [
        "The Intelligent Owner",
        "ZH::The Intelligent Owner",
        "Is the management reasonably efficient?",
        "ZH::Is the management reasonably efficient?",
        "Are the interests of the average outside shareholder receiving proper recognition?",
        "ZH::Are the interests of the average outside shareholder receiving proper recognition?",
        "Body paragraph.",
        "ZH::Body paragraph.",
    ]


def test_epub_invalid_batch_response_falls_back_without_missing_segments(tmp_path):
    epub_path = tmp_path / "fallback.epub"
    _create_epub(epub_path, ["Alpha", "Beta"])
    InvalidBatchModel.reset()

    loader = EPUBBookLoader(
        str(epub_path),
        InvalidBatchModel,
        key="",
        resume=False,
        language="zh-hans",
    )
    loader.exclude_filelist = "nav.xhtml"
    loader.accumulated_num = 400
    loader.make_bilingual_book()

    output_path = tmp_path / "fallback_bilingual.epub"
    assert output_path.exists()
    assert InvalidBatchModel.single_calls == 2
    assert _read_paragraphs(output_path) == [
        "Alpha",
        "ZH::Alpha",
        "Beta",
        "ZH::Beta",
    ]


def test_epub_resume_uses_checkpoint_and_matches_full_run(tmp_path):
    interrupted_path = tmp_path / "resume.epub"
    control_path = tmp_path / "control.epub"
    paragraphs = ["One", "Two", "Three"]
    _create_epub(interrupted_path, paragraphs)
    _create_epub(control_path, paragraphs)

    ResumableModel.reset(interrupt_after=1)
    interrupted_loader = EPUBBookLoader(
        str(interrupted_path),
        ResumableModel,
        key="",
        resume=False,
        language="zh-hans",
    )
    interrupted_loader.exclude_filelist = "nav.xhtml"
    try:
        interrupted_loader.make_bilingual_book()
    except SystemExit:
        pass

    checkpoint_path = tmp_path / ".resume.translation_state.json"
    temp_output_path = tmp_path / "resume_bilingual_temp.epub"
    assert checkpoint_path.exists()
    assert temp_output_path.exists()

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["completed_count"] == 1
    assert len(checkpoint_payload["translations"]) == 1

    ResumableModel.reset()
    resumed_loader = EPUBBookLoader(
        str(interrupted_path),
        ResumableModel,
        key="",
        resume=True,
        language="zh-hans",
    )
    resumed_loader.exclude_filelist = "nav.xhtml"
    resumed_loader.make_bilingual_book()

    ResumableModel.reset()
    control_loader = EPUBBookLoader(
        str(control_path),
        ResumableModel,
        key="",
        resume=False,
        language="zh-hans",
    )
    control_loader.exclude_filelist = "nav.xhtml"
    control_loader.make_bilingual_book()

    assert _read_paragraphs(tmp_path / "resume_bilingual.epub") == _read_paragraphs(
        tmp_path / "control_bilingual.epub"
    )
