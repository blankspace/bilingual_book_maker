import re
import backoff
import logging
from copy import copy

from bs4 import Tag
from bs4.element import NavigableString

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class EPUBBookLoaderHelper:
    def __init__(
        self, translate_model, accumulated_num, translation_style, context_flag
    ):
        self.translate_model = translate_model
        self.accumulated_num = accumulated_num
        self.translation_style = translation_style
        self.context_flag = context_flag

    @staticmethod
    def _normalize_epub_types(tag):
        epub_type = tag.attrs.get("epub:type")
        if epub_type is None:
            return set()
        if isinstance(epub_type, str):
            return {value for value in epub_type.split() if value}
        if isinstance(epub_type, (list, tuple, set)):
            values = set()
            for item in epub_type:
                values.update(str(item).split())
            return values
        return {str(epub_type)}

    @classmethod
    def _is_note_like_tag(cls, tag):
        if not isinstance(tag, Tag):
            return False

        classes = set(tag.get("class", []))
        epub_types = cls._normalize_epub_types(tag)
        return bool(
            {"noteEntry", "footnotetext"} & classes
            or {"rearnote", "rearnotes", "footnote"} & epub_types
        )

    @staticmethod
    def _append_style(tag, extra_style):
        if not extra_style:
            return

        existing_style = tag.get("style", "").strip()
        if existing_style and not existing_style.endswith(";"):
            existing_style = f"{existing_style};"
        extra_style = extra_style.strip()
        if extra_style and not extra_style.endswith(";"):
            extra_style = f"{extra_style};"
        tag["style"] = f"{existing_style} {extra_style}".strip()

    @classmethod
    def _sanitize_translated_tag(cls, source_tag, translated_tag):
        if not isinstance(translated_tag, Tag):
            return

        translated_tag.attrs.pop("id", None)
        translated_tag.attrs.pop("name", None)

        if not cls._is_note_like_tag(source_tag):
            return

        translated_tag.name = "p"
        translated_tag.attrs.pop("epub:type", None)
        translated_tag.attrs.pop("role", None)

        classes = [cls_name for cls_name in translated_tag.get("class", []) if cls_name not in {"noteEntry", "footnotetext"}]
        if classes:
            translated_tag["class"] = classes
        else:
            translated_tag.attrs.pop("class", None)

        cls._append_style(
            translated_tag,
            "margin-left: 0; text-indent: 0; padding-left: 0; width: auto; max-width: none;",
        )

    def insert_trans(self, p, text, translation_style="", single_translate=False):
        if text is None:
            text = ""
        if isinstance(p, NavigableString):
            new_p = NavigableString(text)
            p.insert_after(new_p)
            if single_translate:
                p.extract()
            return
        if (
            p.string is not None
            and p.string.replace(" ", "").strip() == text.replace(" ", "").strip()
        ):
            return
        new_p = copy(p)
        new_p.string = text
        self._sanitize_translated_tag(p, new_p)
        if translation_style != "":
            self._append_style(new_p, translation_style)
        p.insert_after(new_p)
        if single_translate:
            p.extract()

    @backoff.on_exception(
        backoff.expo,
        Exception,
        on_backoff=lambda details: logger.warning(f"retry backoff: {details}"),
        on_giveup=lambda details: logger.warning(f"retry abort: {details}"),
        jitter=None,
    )
    def translate_with_backoff(self, text, context_flag=False):
        return self.translate_model.translate(text, context_flag)

    def deal_new(self, p, wait_p_list, single_translate=False):
        self.deal_old(wait_p_list, single_translate, self.context_flag)
        self.insert_trans(
            p,
            shorter_result_link(self.translate_with_backoff(p.text, self.context_flag)),
            self.translation_style,
            single_translate,
        )

    def deal_old(self, wait_p_list, single_translate=False, context_flag=False):
        if not wait_p_list:
            return

        result_txt_list = self.translate_model.translate_list(wait_p_list)

        for i in range(len(wait_p_list)):
            if i < len(result_txt_list):
                p = wait_p_list[i]
                self.insert_trans(
                    p,
                    shorter_result_link(result_txt_list[i]),
                    self.translation_style,
                    single_translate,
                )

        wait_p_list.clear()


url_pattern = r"(http[s]?://|www\.)+(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"


def is_text_link(text):
    return bool(re.compile(url_pattern).match(text.strip()))


def is_text_tail_link(text, num=80):
    text = text.strip()
    pattern = r".*" + url_pattern + r"$"
    return bool(re.compile(pattern).match(text)) and len(text) < num


def shorter_result_link(text, num=20):
    match = re.search(url_pattern, text)

    if not match or len(match.group()) < num:
        return text

    return re.compile(url_pattern).sub("...", text)


def is_text_source(text):
    return text.strip().startswith("Source: ")


def is_text_list(text, num=80):
    text = text.strip()
    return re.match(r"^Listing\s*\d+", text) and len(text) < num


def is_text_figure(text, num=80):
    text = text.strip()
    return re.match(r"^Figure\s*\d+", text) and len(text) < num


def is_text_digit_and_space(s):
    for c in s:
        if not c.isdigit() and not c.isspace():
            return False
    return True


def is_text_isbn(s):
    pattern = r"^[Ee]?ISBN\s*\d[\d\s]*$"
    return bool(re.match(pattern, s))


def not_trans(s):
    return any(
        [
            is_text_link(s),
            is_text_tail_link(s),
            is_text_source(s),
            is_text_list(s),
            is_text_figure(s),
            is_text_digit_and_space(s),
            is_text_isbn(s),
        ]
    )
