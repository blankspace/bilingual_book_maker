from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from book_maker.loader.pdf_loader import MOBIBookLoader, PDFBookLoader


class DummyModel:
    translate_calls = 0
    translate_list_calls = 0
    interrupt_after_calls = None

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
        self.model = "dummy-model"
        self.prompt_template = "Translate {text} to {language}"
        self.prompt_sys_msg = ""

    @classmethod
    def reset(cls, interrupt_after_calls=None):
        cls.translate_calls = 0
        cls.translate_list_calls = 0
        cls.interrupt_after_calls = interrupt_after_calls

    def translate(self, text):
        type(self).translate_calls += 1
        if (
            type(self).interrupt_after_calls is not None
            and type(self).translate_calls > type(self).interrupt_after_calls
        ):
            raise KeyboardInterrupt()
        return f"<T>{text}"

    def translate_list(self, texts):
        type(self).translate_list_calls += 1
        return [self.translate(text) for text in texts]


def _create_pdf(pdf_path, page_lines):
    doc = fitz.open()
    for lines in page_lines:
        page = doc.new_page()
        for line_index, line in enumerate(lines):
            page.insert_text((72, 72 + (line_index * 18)), line)
    doc.save(str(pdf_path))


def test_pdf_loader_extracts_and_translates(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _create_pdf(pdf_path, [["Hello world", "This is a PDF test"]])
    DummyModel.reset()

    loader = PDFBookLoader(
        str(pdf_path),
        DummyModel,
        key="",
        resume=False,
        language="en",
        is_test=True,
        test_num=5,
    )

    assert len(loader.origin_book) > 0

    loader.make_bilingual_book()

    out_file = tmp_path / "test_bilingual.txt"
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    content = out_file.read_text(encoding="utf-8")
    assert "<T>Hello world" in content
    assert "<T>This is a PDF test" in content

    epub_file = tmp_path / "test_bilingual.epub"
    assert epub_file.exists()
    assert epub_file.stat().st_size > 0


def test_pdf_loader_resume_uses_json_checkpoint(tmp_path):
    pdf_path = tmp_path / "resume.pdf"
    _create_pdf(pdf_path, [["First block"], ["Second block"], ["Third block"]])
    DummyModel.reset(interrupt_after_calls=1)

    loader = PDFBookLoader(
        str(pdf_path),
        DummyModel,
        key="",
        resume=False,
        language="en",
    )
    loader.batch_size = 1

    with pytest.raises(SystemExit):
        loader.make_bilingual_book()

    checkpoint_path = tmp_path / ".resume.translation_state.json"
    assert checkpoint_path.exists()
    assert (tmp_path / "resume_bilingual_temp.txt").exists()
    assert (tmp_path / "resume_bilingual_temp.epub").exists()

    DummyModel.reset()
    resumed_loader = PDFBookLoader(
        str(pdf_path),
        DummyModel,
        key="",
        resume=True,
        language="en",
    )
    resumed_loader.batch_size = 1
    resumed_loader.make_bilingual_book()

    final_content = (tmp_path / "resume_bilingual.txt").read_text(encoding="utf-8")
    assert "<T>First block" in final_content
    assert "<T>Second block" in final_content
    assert "<T>Third block" in final_content


def test_mobi_loader_uses_fitz_pipeline(monkeypatch, tmp_path):
    mobi_path = tmp_path / "sample.mobi"
    mobi_path.write_bytes(b"fake-mobi")

    class FakePage:
        def __init__(self, blocks):
            self._blocks = blocks

        def get_text(self, kind):
            assert kind == "blocks"
            return self._blocks

    class FakeDocument:
        def __init__(self):
            self._pages = [
                FakePage([(0, 0, 100, 20, "Chapter One\nHello world", 0, 0)]),
                FakePage([(0, 0, 100, 20, "Second page paragraph", 0, 0)]),
            ]

        def __iter__(self):
            return iter(self._pages)

        def close(self):
            pass

    monkeypatch.setattr(
        "book_maker.loader.pdf_loader.fitz.open",
        lambda path: FakeDocument(),
    )
    DummyModel.reset()

    loader = MOBIBookLoader(
        str(mobi_path),
        DummyModel,
        key="",
        resume=False,
        language="en",
    )
    loader.make_bilingual_book()

    out_file = tmp_path / "sample_bilingual.txt"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<T>Chapter One Hello world" in content
    assert "<T>Second page paragraph" in content
