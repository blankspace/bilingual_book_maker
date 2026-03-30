import os
import pickle
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import traceback
from threading import Lock
from typing import Any

from bs4 import BeautifulSoup as bs
from bs4 import Tag
from bs4.element import NavigableString
from ebooklib import ITEM_DOCUMENT, epub
from rich import print
from tqdm import tqdm

from book_maker.utils import num_tokens_from_text, prompt_config_to_kwargs

from .base_loader import BaseBookLoader
from .helper import EPUBBookLoaderHelper, is_text_link, not_trans

DEFAULT_TRANSLATE_TAGS = "auto"
AUTO_TRANSLATE_TAGS = (
    "p",
    "div",
    "aside",
    "section",
    "article",
    "li",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "figcaption",
    "caption",
    "td",
    "th",
    "dt",
    "dd",
)
AUTO_EXCLUDE_TRANSLATE_TAGS = ("pre", "code", "script", "style", "svg", "math")


@dataclass
class TranslationSegment:
    segment_id: str
    item_file_name: str
    ordinal: int
    text: str
    node: Any
    replace_target: bool = False


@dataclass
class TranslationCheckpoint:
    source_path: str
    source_sha256: str
    config: dict[str, Any]
    config_fingerprint: str
    translations: dict[str, str] = field(default_factory=dict)
    committed_batches: int = 0
    completed_count: int = 0

    def to_dict(self):
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "translations": self.translations,
            "committed_batches": self.committed_batches,
            "completed_count": self.completed_count,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            source_path=payload["source_path"],
            source_sha256=payload["source_sha256"],
            config=payload["config"],
            config_fingerprint=payload["config_fingerprint"],
            translations=dict(payload.get("translations", {})),
            committed_batches=payload.get("committed_batches", 0),
            completed_count=payload.get("completed_count", 0),
        )


class EPUBBookLoader(BaseBookLoader):
    def __init__(
        self,
        epub_name,
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
    ):
        self.epub_name = epub_name
        self.new_epub = epub.EpubBook()
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
        self.is_test = is_test
        self.test_num = test_num
        self.translate_tags = DEFAULT_TRANSLATE_TAGS
        self.exclude_translate_tags = "sup"
        self.allow_navigable_strings = False
        self.accumulated_num = 1
        self.translation_style = ""
        self.context_flag = context_flag
        self.helper = EPUBBookLoaderHelper(
            self.translate_model,
            self.accumulated_num,
            self.translation_style,
            self.context_flag,
        )
        self.retranslate = None
        self.exclude_filelist = ""
        self.only_filelist = ""
        self.single_translate = single_translate
        self.block_size = 1  # Default to 1 for better translation quality with delimiter-based batching
        self.batch_use_flag = False
        self.batch_flag = False
        self.parallel_workers = 1
        self.enable_parallel = False
        self._progress_lock = Lock()
        self._translation_index = 0
        self._legacy_state_loaded = False
        self._checkpoint_state = None
        self.prompt_config = prompt_config or {}
        self.bin_path = f"{Path(epub_name).parent}/.{Path(epub_name).stem}.temp.bin"
        self.checkpoint_path = (
            f"{Path(epub_name).parent}/.{Path(epub_name).stem}.translation_state.json"
        )
        self.set_parallel_workers(parallel_workers)

        # monkey patch for # 173
        def _write_items_patch(obj):
            for item in obj.book.get_items():
                if isinstance(item, epub.EpubNcx):
                    obj.out.writestr(
                        "%s/%s" % (obj.book.FOLDER_NAME, item.file_name), obj._get_ncx()
                    )
                elif isinstance(item, epub.EpubNav):
                    obj.out.writestr(
                        "%s/%s" % (obj.book.FOLDER_NAME, item.file_name),
                        obj._get_nav(item),
                    )
                elif item.manifest:
                    obj.out.writestr(
                        "%s/%s" % (obj.book.FOLDER_NAME, item.file_name), item.content
                    )
                else:
                    obj.out.writestr("%s" % item.file_name, item.content)

        def _check_deprecated(obj):
            pass

        epub.EpubWriter._write_items = _write_items_patch
        epub.EpubReader._check_deprecated = _check_deprecated

        try:
            self.origin_book = epub.read_epub(self.epub_name)
        except Exception:
            # tricky monkey patch for #71 if you don't know why please check the issue and ignore this
            # when upstream change will TODO fix this
            def _load_spine(obj):
                spine = obj.container.find("{%s}%s" % (epub.NAMESPACES["OPF"], "spine"))

                obj.book.spine = [
                    (t.get("idref"), t.get("linear", "yes")) for t in spine
                ]
                obj.book.set_direction(spine.get("page-progression-direction", None))

            epub.EpubReader._load_spine = _load_spine
            self.origin_book = epub.read_epub(self.epub_name)

        self.p_to_save = []
        self.resume = resume

    @staticmethod
    def _is_special_text(text):
        return (
            text.isdigit()
            or text.isspace()
            or is_text_link(text)
            or all(char in string.punctuation for char in text)
        )

    def _make_new_book(self, book):
        new_book = epub.EpubBook()
        allowed_ns = set(epub.NAMESPACES.keys()) | set(epub.NAMESPACES.values())

        for namespace, metas in book.metadata.items():
            # Only keep namespaces recognized by ebooklib
            if namespace not in allowed_ns:
                continue

            if isinstance(metas, dict):
                entries = (
                    (name, value, others)
                    for name, values in metas.items()
                    for value, others in (
                        (item if isinstance(item, tuple) else (item, None))
                        for item in values
                    )
                )
            else:
                entries = metas

            for entry in entries:
                if not entry:
                    continue

                if isinstance(entry, tuple):
                    if len(entry) == 3:
                        name, value, others = entry
                    elif len(entry) == 2:
                        name, value = entry
                        others = None
                    else:
                        continue
                else:
                    # Unexpected metadata format; skip gracefully
                    continue

                # `others` can be {} or None
                if others:
                    new_book.add_metadata(namespace, name, value, others)
                else:
                    new_book.add_metadata(namespace, name, value)

        new_book.spine = book.spine
        new_book.toc = self._fix_toc_uids(book.toc)
        return new_book

    def _fix_toc_uids(self, toc, counter=None):
        """Fix TOC items that have uid=None to prevent TypeError when writing NCX."""
        if counter is None:
            counter = [0]  # Use list to allow mutation in nested calls

        fixed_toc = []
        for item in toc:
            if isinstance(item, tuple):
                # Section with sub-items: (Section, [sub-items])
                section, sub_items = item
                if hasattr(section, "uid") and section.uid is None:
                    section.uid = f"navpoint-{counter[0]}"
                    counter[0] += 1
                fixed_sub_items = self._fix_toc_uids(sub_items, counter)
                fixed_toc.append((section, fixed_sub_items))
            elif hasattr(item, "uid"):
                # Link or EpubHtml item
                if item.uid is None:
                    item.uid = f"navpoint-{counter[0]}"
                    counter[0] += 1
                fixed_toc.append(item)
            else:
                fixed_toc.append(item)

        return fixed_toc

    @staticmethod
    def _normalize_tag_csv(tag_csv):
        return [tag.strip().lower() for tag in tag_csv.split(",") if tag.strip()]

    def _is_auto_translate_tags(self):
        configured_tags = self._normalize_tag_csv(self.translate_tags)
        if not configured_tags:
            return True
        return any(tag in {"auto", "all", "*"} for tag in configured_tags)

    def _get_translate_tag_names(self):
        if self._is_auto_translate_tags():
            return list(AUTO_TRANSLATE_TAGS)
        return self._normalize_tag_csv(self.translate_tags)

    def _get_exclude_translate_tag_names(self):
        exclude_tags = self._normalize_tag_csv(self.exclude_translate_tags)
        if self._is_auto_translate_tags():
            exclude_tags.extend(AUTO_EXCLUDE_TRANSLATE_TAGS)
        return list(dict.fromkeys(exclude_tags))

    def _should_use_checkpoint_pipeline(self):
        return not any(
            [
                self.retranslate,
                self.batch_flag,
                self.batch_use_flag,
                self.single_translate and self.block_size > 0,
            ]
        )

    def _compute_source_sha256(self):
        digest = hashlib.sha256()
        with open(self.epub_name, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_runtime_config(self):
        prompt_template = getattr(self.translate_model, "prompt_template", None)
        if prompt_template is None:
            prompt_template = getattr(self.translate_model, "prompt", "")

        prompt_sys_msg = getattr(self.translate_model, "prompt_sys_msg", "")
        runtime_config = {
            "language": self.translate_model.language,
            "translator": self.translate_model.__class__.__name__,
            "selected_model_name": getattr(self, "selected_model_name", None),
            "translator_model": getattr(self.translate_model, "model", None),
            "prompt_template": prompt_template,
            "prompt_sys_msg": prompt_sys_msg,
            "translate_tags_mode": self.translate_tags,
            "translate_tags": self._get_translate_tag_names(),
            "exclude_translate_tags_mode": self.exclude_translate_tags,
            "exclude_translate_tags": self._get_exclude_translate_tag_names(),
            "allow_navigable_strings": self.allow_navigable_strings,
            "single_translate": self.single_translate,
            "accumulated_num": self.accumulated_num,
            "context_flag": self.context_flag,
        }
        runtime_payload = json.dumps(
            runtime_config, sort_keys=True, ensure_ascii=False
        )
        return runtime_config, hashlib.sha256(runtime_payload.encode("utf-8")).hexdigest()

    def _default_checkpoint_state(self):
        config, config_fingerprint = self._build_runtime_config()
        return TranslationCheckpoint(
            source_path=str(Path(self.epub_name).resolve()),
            source_sha256=self._compute_source_sha256(),
            config=config,
            config_fingerprint=config_fingerprint,
        )

    def _load_json_checkpoint_state(self):
        checkpoint = self._default_checkpoint_state()
        if not self.resume:
            self._checkpoint_state = checkpoint
            return checkpoint

        if not os.path.exists(self.checkpoint_path):
            self._checkpoint_state = checkpoint
            return checkpoint

        with open(self.checkpoint_path, encoding="utf-8") as f:
            payload = json.load(f)
        loaded = TranslationCheckpoint.from_dict(payload)

        if loaded.source_sha256 != checkpoint.source_sha256:
            raise ValueError("resume checkpoint does not match the current EPUB file")
        if loaded.config_fingerprint != checkpoint.config_fingerprint:
            raise ValueError("resume checkpoint does not match the current translation parameters")

        self._checkpoint_state = loaded
        return loaded

    def _load_legacy_progress_into_checkpoint(self):
        if self.accumulated_num > 1 or not os.path.exists(self.bin_path):
            return

        if self._legacy_state_loaded:
            return

        try:
            with open(self.bin_path, "rb") as f:
                self.p_to_save = pickle.load(f)
                self._legacy_state_loaded = True
        except Exception:
            return

    def _restore_context_pair(self, source_text, translated_text):
        if not getattr(self.translate_model, "context_flag", False):
            return
        save_context = getattr(self.translate_model, "save_context", None)
        if callable(save_context):
            save_context(source_text, translated_text)

    def _get_filtered_translatable_tags(self, root, trans_taglist):
        return self.filter_nest_list(root.find_all(trans_taglist), trans_taglist)

    def _build_table_translation_target_map(self, table, trans_taglist):
        translation_table = copy(table)
        table.insert_after(translation_table)

        source_nodes = self._get_filtered_translatable_tags(table, trans_taglist)
        target_nodes = self._get_filtered_translatable_tags(
            translation_table, trans_taglist
        )
        if len(source_nodes) != len(target_nodes):
            return {}

        return {id(source): target for source, target in zip(source_nodes, target_nodes)}

    def _resolve_translation_target(self, node, trans_taglist, table_target_maps):
        if not isinstance(node, Tag):
            return node, False

        table = node.find_parent("table")
        if table is None:
            return node, False

        table_key = id(table)
        if table_key not in table_target_maps:
            table_target_maps[table_key] = self._build_table_translation_target_map(
                table, trans_taglist
            )

        target_node = table_target_maps[table_key].get(id(node))
        if target_node is None:
            return node, False
        return target_node, True

    def _create_segment(
        self, item, ordinal, node, target_node=None, replace_target=False
    ):
        if isinstance(node, NavigableString):
            text = str(node)
        else:
            text = self._extract_paragraph(copy(node)).get_text()

        if not text:
            return None
        if self._is_special_text(text) or not_trans(text):
            return None

        return TranslationSegment(
            segment_id=f"{item.file_name}#{ordinal}",
            item_file_name=item.file_name,
            ordinal=ordinal,
            text=text,
            node=target_node or node,
            replace_target=replace_target,
        )

    def _build_document_context(self, item, max_segments=None):
        if self.only_filelist and item.file_name not in self.only_filelist.split(","):
            return {"item": item, "soup": None, "segments": [], "skip": True}
        if not self.only_filelist and item.file_name in self.exclude_filelist.split(","):
            return {"item": item, "soup": None, "segments": [], "skip": True}

        soup = bs(item.content, "html.parser")
        trans_taglist = self._get_translate_tag_names()
        p_list = soup.find_all(trans_taglist)
        p_list = self.filter_nest_list(p_list, trans_taglist)

        if self.allow_navigable_strings:
            p_list.extend(soup.find_all(string=True))

        if max_segments == 0:
            return {"item": item, "soup": soup, "segments": [], "skip": False}

        segments = []
        ordinal = 0
        table_target_maps = {}
        for node in p_list:
            ordinal += 1
            segment = self._create_segment(item, ordinal, node)
            if segment is None:
                continue

            target_node, replace_target = self._resolve_translation_target(
                node, trans_taglist, table_target_maps
            )
            if replace_target:
                segment.node = target_node
                segment.replace_target = True

            segments.append(segment)
            if max_segments is not None and len(segments) >= max_segments:
                break

        return {"item": item, "soup": soup, "segments": segments, "skip": False}

    def _collect_document_contexts(self):
        contexts = []
        remaining = self.test_num if self.is_test else None
        for item in self.origin_book.get_items_of_type(ITEM_DOCUMENT):
            max_segments = remaining if remaining is not None else None
            context = self._build_document_context(item, max_segments=max_segments)
            contexts.append(context)
            if remaining is not None and not context["skip"]:
                remaining -= len(context["segments"])
                if remaining < 0:
                    remaining = 0
        return contexts

    def _iter_segment_batches(self, segments):
        if self.accumulated_num <= 1:
            for segment in segments:
                yield [segment]
            return

        current_batch = []
        current_tokens = 0
        max_segments = 8

        for segment in segments:
            segment_tokens = num_tokens_from_text(segment.text)
            if not current_batch:
                current_batch = [segment]
                current_tokens = segment_tokens
                continue

            exceeds_token_limit = current_tokens + segment_tokens > self.accumulated_num
            exceeds_batch_limit = len(current_batch) >= max_segments
            if exceeds_token_limit or exceeds_batch_limit:
                yield current_batch
                current_batch = [segment]
                current_tokens = segment_tokens
            else:
                current_batch.append(segment)
                current_tokens += segment_tokens

        if current_batch:
            yield current_batch

    def _validate_segment_translations(self, segments, translated_segments):
        if not isinstance(translated_segments, list):
            raise ValueError("translated segments must be a list")

        expected_ids = [segment.segment_id for segment in segments]
        seen_ids = set()
        normalized = {}

        for translated_segment in translated_segments:
            if not isinstance(translated_segment, dict):
                raise ValueError("each translated segment must be an object")
            segment_id = translated_segment.get("id")
            translation = translated_segment.get("translation")
            if not isinstance(segment_id, str):
                raise ValueError("translated segment id must be a string")
            if segment_id in seen_ids:
                raise ValueError(f"duplicate translated segment id: {segment_id}")
            if translation is None:
                raise ValueError(f"translated segment {segment_id} is missing translation")
            seen_ids.add(segment_id)
            normalized[segment_id] = str(translation)

        if set(expected_ids) != set(normalized):
            raise ValueError("translated segment ids do not match requested segments")

        return {segment.segment_id: normalized[segment.segment_id] for segment in segments}

    def _translate_single_segment(self, segment):
        translated_text = self.translate_model.translate(segment.text)
        if translated_text is None:
            raise RuntimeError(
                f"segment translation returned None for {segment.segment_id}"
            )
        return {segment.segment_id: translated_text}

    def _translate_batch_with_fallback(self, segments, retry_allowed=True):
        try:
            translated_segments = self.translate_model.translate_segments(
                [{"id": segment.segment_id, "text": segment.text} for segment in segments]
            )
            return self._validate_segment_translations(segments, translated_segments)
        except KeyboardInterrupt:
            raise
        except Exception:
            if retry_allowed:
                return self._translate_batch_with_fallback(segments, retry_allowed=False)
            if len(segments) == 1:
                return self._translate_single_segment(segments[0])
            midpoint = len(segments) // 2
            left = self._translate_batch_with_fallback(segments[:midpoint], retry_allowed=False)
            right = self._translate_batch_with_fallback(segments[midpoint:], retry_allowed=False)
            merged = {}
            merged.update(left)
            merged.update(right)
            return merged

    @staticmethod
    def _replace_translation_target(node, translated_text):
        if translated_text is None:
            translated_text = ""

        if isinstance(node, NavigableString):
            node.replace_with(NavigableString(translated_text))
            return

        node.clear()
        node.append(NavigableString(translated_text))

    def _apply_translation_to_segment(self, segment, translated_text):
        if segment.replace_target:
            self._replace_translation_target(segment.node, translated_text)
            return

        self.helper.insert_trans(
            segment.node,
            translated_text,
            self.translation_style,
            self.single_translate,
        )

    def _save_json_checkpoint(self):
        if self._checkpoint_state is None:
            return

        target_path = Path(self.checkpoint_path)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                self._checkpoint_state.to_dict(),
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        os.replace(temp_path, target_path)

    def _commit_batch(self, batch, batch_translations, pbar):
        for segment in batch:
            translated_text = batch_translations[segment.segment_id]
            self._checkpoint_state.translations[segment.segment_id] = translated_text
            self._apply_translation_to_segment(segment, translated_text)
            self._checkpoint_state.completed_count += 1
            pbar.update(1)

        self._checkpoint_state.committed_batches += 1
        self._save_json_checkpoint()

    def _replay_saved_translations(self, context, pbar):
        missing_segments = []
        for segment in context["segments"]:
            translated_text = self._checkpoint_state.translations.get(segment.segment_id)
            if translated_text is None:
                missing_segments.append(segment)
                continue

            self._apply_translation_to_segment(segment, translated_text)
            self._restore_context_pair(segment.text, translated_text)
            pbar.update(1)

        return missing_segments

    def _process_context_with_checkpoint(self, context, pbar):
        if context["skip"] or not context["segments"]:
            return

        missing_segments = self._replay_saved_translations(context, pbar)
        for batch in self._iter_segment_batches(missing_segments):
            batch_translations = self._translate_batch_with_fallback(batch)
            self._commit_batch(batch, batch_translations, pbar)

        if context["soup"] is not None:
            context["item"].content = context["soup"].encode(encoding="utf-8")

    def _seed_checkpoint_from_legacy_progress(self, contexts):
        if self._checkpoint_state.translations or not self.p_to_save:
            return

        translated_iter = iter(self.p_to_save)
        for context in contexts:
            for segment in context["segments"]:
                translated_text = next(translated_iter, None)
                if translated_text is None:
                    return
                self._checkpoint_state.translations[segment.segment_id] = translated_text
                self._checkpoint_state.completed_count += 1

    def _make_bilingual_book_with_checkpoint(self):
        self._checkpoint_state = self._load_json_checkpoint_state()
        contexts = self._collect_document_contexts()
        self._load_legacy_progress_into_checkpoint()
        self._seed_checkpoint_from_legacy_progress(contexts)

        new_book = self._make_new_book(self.origin_book)
        for item in self.origin_book.get_items():
            if item.get_type() != ITEM_DOCUMENT:
                new_book.add_item(item)

        total_segments = sum(len(context["segments"]) for context in contexts)
        pbar = tqdm(total=self.test_num) if self.is_test else tqdm(total=total_segments)

        try:
            for context in contexts:
                self._process_context_with_checkpoint(context, pbar)
                new_book.add_item(context["item"])

            name, _ = os.path.splitext(self.epub_name)
            epub.write_epub(f"{name}_bilingual.epub", new_book, {})
            pbar.close()
        except KeyboardInterrupt as e:
            print(e)
            print("you can resume it next time")
            self._save_json_checkpoint()
            self._save_temp_book()
            if total_segments:
                pbar.close()
            sys.exit(0)
        except Exception:
            traceback.print_exc()
            self._save_json_checkpoint()
            self._save_temp_book()
            if total_segments:
                pbar.close()
            sys.exit(0)

    def _extract_paragraph(self, p):
        for p_exclude in self._get_exclude_translate_tag_names():
            # for issue #280
            if type(p) is NavigableString:
                continue
            for pt in p.find_all(p_exclude):
                pt.extract()
        return p

    def _process_paragraph(self, p, new_p, index, p_to_save_len, thread_safe=False):
        if self.resume and index < p_to_save_len:
            # When resuming, keep original text in p, only restore translation
            # p.string should remain as original text from source EPUB
            new_p.string = self.p_to_save[index]
        else:
            t_text = ""
            if self.batch_flag:
                self.translate_model.add_to_batch_translate_queue(index, new_p.text)
            elif self.batch_use_flag:
                t_text = self.translate_model.batch_translate(index)
            else:
                t_text = self.translate_model.translate(new_p.text)
            if t_text is None:
                raise RuntimeError(
                    "`t_text` is None: your translation model is not working as expected. Please check your translation model configuration."
                )
            if type(p) is NavigableString:
                new_p = t_text
                self.p_to_save.append(new_p)
            else:
                new_p.string = t_text
                self.p_to_save.append(new_p.text)

        self.helper.insert_trans(
            p, new_p.string, self.translation_style, self.single_translate
        )
        index += 1

        if thread_safe:
            with self._progress_lock:
                if index % 20 == 0:
                    self._save_progress()
        else:
            if index % 20 == 0:
                self._save_progress()
        return index

    def _process_combined_paragraph(
        self, p_block, index, p_to_save_len, thread_safe=False
    ):
        """Returns (new_index, processed_count)."""
        text = []
        text_paragraphs = []  # Track which paragraphs correspond to text[]
        translated_cache = []  # Cache translated text for resumed paragraphs
        processed_count = 0

        for p in p_block:
            if self.is_test and index >= self.test_num:
                break

            if self.resume and index < p_to_save_len:
                # When resuming, cache the translation but don't modify p yet
                translated_cache.append(self.p_to_save[index])
            else:
                p_text = p.text.rstrip()
                text.append(p_text)
                text_paragraphs.append(p)

            index += 1
            processed_count += 1

        if len(text) > 0:
            # Use translate_list with delimiter for proper paragraph separation
            try:
                translated_text_list = self.translate_model.translate_list(text)
            except Exception as e:
                print(f"[bold red]Translation error: {str(e)}[/bold red]")
                raise

            for i, t in enumerate(translated_text_list):
                p = (
                    text_paragraphs[i]
                    if i < len(text_paragraphs)
                    else text_paragraphs[-1]
                )
                self.helper.insert_trans(
                    p, t, self.translation_style, self.single_translate
                )
                print(text[i])
                print(f"[bold green]{t}[/bold green]")
                print()
        else:
            # Handle resumed paragraphs - insert translations without modifying originals
            for i, p in enumerate(p_block):
                if i < len(translated_cache):
                    new_p = copy(p)
                    new_p.string = translated_cache[i]
                    self.helper.insert_trans(
                        p, new_p.string, self.translation_style, self.single_translate
                    )

        if thread_safe:
            with self._progress_lock:
                self._save_progress()
        else:
            self._save_progress()
        return index, processed_count

    def translate_paragraphs_acc(self, p_list, send_num):
        count = 0
        wait_p_list = []
        for i in range(len(p_list)):
            p = p_list[i]
            print(f"translating {i}/{len(p_list)}")
            temp_p = copy(p)

            for p_exclude in self._get_exclude_translate_tag_names():
                # for issue #280
                if type(p) is NavigableString:
                    continue
                for pt in temp_p.find_all(p_exclude):
                    pt.extract()

            if any(
                [not p.text, self._is_special_text(temp_p.text), not_trans(temp_p.text)]
            ):
                if i == len(p_list) - 1:
                    self.helper.deal_old(wait_p_list, self.single_translate)
                continue
            length = num_tokens_from_text(temp_p.text)
            if length > send_num:
                self.helper.deal_new(p, wait_p_list, self.single_translate)
                continue
            if i == len(p_list) - 1:
                if count + length < send_num:
                    wait_p_list.append(p)
                    self.helper.deal_old(wait_p_list, self.single_translate)
                else:
                    self.helper.deal_new(p, wait_p_list, self.single_translate)
                break
            if count + length < send_num:
                count += length
                wait_p_list.append(p)
            else:
                self.helper.deal_old(wait_p_list, self.single_translate)
                wait_p_list.append(p)
                count = length

    def get_item(self, book, name):
        for item in book.get_items():
            if item.file_name == name:
                return item

    def find_items_containing_string(self, book, search_string):
        matching_items = []

        for item in book.get_items_of_type(ITEM_DOCUMENT):
            content = item.get_content()
            soup = bs(content, "html.parser")
            if search_string in soup.get_text():
                matching_items.append(item)

        return matching_items

    def retranslate_book(self, index, p_to_save_len, pbar, trans_taglist, retranslate):
        complete_book_name = retranslate[0]
        fixname = retranslate[1]
        fixstart = retranslate[2]
        fixend = retranslate[3]

        if fixend == "":
            fixend = fixstart

        name_fix = complete_book_name

        complete_book = epub.read_epub(complete_book_name)

        if fixname == "":
            fixname = self.find_items_containing_string(complete_book, fixstart)[
                0
            ].file_name
            print(f"auto find fixname: {fixname}")

        new_book = self._make_new_book(complete_book)

        complete_item = self.get_item(complete_book, fixname)
        if complete_item is None:
            return

        ori_item = self.get_item(self.origin_book, fixname)
        if ori_item is None:
            return

        content_complete = complete_item.content
        content_ori = ori_item.content
        soup_complete = bs(content_complete, "html.parser")
        soup_ori = bs(content_ori, "html.parser")

        p_list_complete = soup_complete.find_all(trans_taglist)
        p_list_ori = soup_ori.find_all(trans_taglist)

        target = None
        tagl = []

        # extract from range
        find_end = False
        find_start = False
        for tag in p_list_complete:
            if find_end:
                tagl.append(tag)
                break

            if fixend in tag.text:
                find_end = True
            if fixstart in tag.text:
                find_start = True

            if find_start:
                if not target:
                    target = tag.previous_sibling
                tagl.append(tag)

        for t in tagl:
            t.extract()

        flag = False
        extract_p_list_ori = []
        for p in p_list_ori:
            if fixstart in p.text:
                flag = True
            if flag:
                extract_p_list_ori.append(p)
            if fixend in p.text:
                break

        for t in extract_p_list_ori:
            if target:
                target.insert_after(t)
                target = t

        for item in complete_book.get_items():
            if item.file_name != fixname:
                new_book.add_item(item)
        if soup_complete:
            complete_item.content = soup_complete.encode()

        index = self.process_item(
            complete_item,
            index,
            p_to_save_len,
            pbar,
            new_book,
            trans_taglist,
            fixstart,
            fixend,
        )
        epub.write_epub(f"{name_fix}", new_book, {})

    def has_nest_child(self, element, trans_taglist):
        if isinstance(element, Tag):
            for child in element.children:
                if child.name in trans_taglist:
                    return True
                if self.has_nest_child(child, trans_taglist):
                    return True
        return False

    def filter_nest_list(self, p_list, trans_taglist):
        filtered_list = [p for p in p_list if not self.has_nest_child(p, trans_taglist)]
        return filtered_list

    def process_item(
        self,
        item,
        index,
        p_to_save_len,
        pbar,
        new_book,
        trans_taglist,
        fixstart=None,
        fixend=None,
    ):
        if self.only_filelist != "" and item.file_name not in self.only_filelist.split(
            ","
        ):
            return index
        elif self.only_filelist == "" and item.file_name in self.exclude_filelist.split(
            ","
        ):
            new_book.add_item(item)
            return index

        if not os.path.exists("log"):
            os.makedirs("log")

        content = item.content
        soup = bs(content, "html.parser")
        p_list = soup.find_all(trans_taglist)

        p_list = self.filter_nest_list(p_list, trans_taglist)

        if self.retranslate:
            new_p_list = []

            if fixstart is None or fixend is None:
                return

            start_append = False
            for p in p_list:
                text = p.get_text()
                if fixstart in text or fixend in text or start_append:
                    start_append = True
                    new_p_list.append(p)
                if fixend in text:
                    p_list = new_p_list
                    break

        if self.allow_navigable_strings:
            p_list.extend(soup.find_all(string=True))

        send_num = self.accumulated_num
        if send_num > 1:
            with open("log/buglog.txt", "a") as f:
                print(f"------------- {item.file_name} -------------", file=f)

            print("------------------------------------------------------")
            print(f"dealing {item.file_name} ...")
            self.translate_paragraphs_acc(p_list, send_num)
        else:
            is_test_done = self.is_test and index >= self.test_num
            p_block = []
            block_len = 0
            for p in p_list:
                if is_test_done:
                    break
                if not p.text or self._is_special_text(p.text):
                    # Skip empty/special paragraphs without updating progress bar
                    continue

                new_p = self._extract_paragraph(copy(p))
                if self.block_size >= 1:
                    # Collect paragraphs for batch translation
                    p_block.append(p)

                    # Process when we have enough paragraphs
                    if len(p_block) >= self.block_size:
                        index, n = self._process_combined_paragraph(
                            p_block, index, p_to_save_len, thread_safe=False
                        )
                        pbar.update(n)
                        p_block = []
                        print()
                else:
                    index = self._process_paragraph(
                        p, new_p, index, p_to_save_len, thread_safe=False
                    )
                    print()
                    pbar.update(1)

                if self.is_test and index >= self.test_num:
                    is_test_done = True
                    break

            # Process remaining paragraphs in the batch
            if self.block_size >= 1 and len(p_block) > 0:
                index, n = self._process_combined_paragraph(
                    p_block, index, p_to_save_len, thread_safe=False
                )
                pbar.update(n)

        if soup:
            item.content = soup.encode(encoding="utf-8")
        new_book.add_item(item)

        return index

    def set_parallel_workers(self, workers):
        """Set number of parallel workers for chapter processing.

        Args:
            workers (int): Number of parallel workers. Will be automatically
                         optimized based on actual chapter count during processing.
        """
        self.parallel_workers = max(1, workers)
        self.enable_parallel = workers > 1

        if workers > 8:
            print(
                f"⚠️  Warning: {workers} workers is quite high. Consider using 2-8 workers for optimal performance."
            )

    def _get_next_translation_index(self):
        """Thread-safe method to get next translation index."""
        with self._progress_lock:
            index = self._translation_index
            self._translation_index += 1
            return index

    def _process_chapter_parallel(self, chapter_data):
        """Process a single chapter in parallel mode with proper accumulated_num handling."""
        item, trans_taglist, p_to_save_len = chapter_data
        chapter_result = {
            "item": item,
            "processed_content": None,
            "success": False,
            "error": None,
        }

        try:
            # Create a chapter-specific translator instance to avoid context conflicts
            # This ensures each chapter has its own independent context
            thread_translator = self._create_chapter_translator()

            content = item.content
            soup = bs(content, "html.parser")
            p_list = soup.find_all(trans_taglist)
            p_list = self.filter_nest_list(p_list, trans_taglist)

            if self.allow_navigable_strings:
                p_list.extend(soup.find_all(string=True))

            # Initialize chapter-specific context lists
            chapter_context_list = []
            chapter_translated_list = []

            # Apply accumulated_num logic for this chapter independently
            send_num = self.accumulated_num
            if send_num > 1:
                # Use accumulated translation logic for this chapter
                self._translate_paragraphs_acc_parallel(
                    p_list,
                    send_num,
                    thread_translator,
                    chapter_context_list,
                    chapter_translated_list,
                )
            else:
                # Process paragraphs individually for this chapter
                for p in p_list:
                    if not p.text or self._is_special_text(p.text):
                        continue

                    new_p = self._extract_paragraph(copy(p))
                    index = self._get_next_translation_index()

                    if self.resume and index < p_to_save_len:
                        t_text = self.p_to_save[index]
                    else:
                        # Use chapter-specific context for translation
                        t_text = self._translate_with_chapter_context(
                            thread_translator,
                            new_p.text,
                            chapter_context_list,
                            chapter_translated_list,
                        )
                        t_text = "" if t_text is None else t_text
                        with self._progress_lock:
                            self.p_to_save.append(t_text)

                    if isinstance(p, NavigableString):
                        translated_node = NavigableString(t_text)
                        p.insert_after(translated_node)
                        if self.single_translate:
                            p.extract()
                    else:
                        self.helper.insert_trans(
                            p, t_text, self.translation_style, self.single_translate
                        )

                    with self._progress_lock:
                        if index % 20 == 0:
                            self._save_progress()

            if soup:
                chapter_result["processed_content"] = soup.encode(encoding="utf-8")
            chapter_result["success"] = True

        except Exception as e:
            chapter_result["error"] = str(e)
            print(f"Error processing chapter {item.file_name}: {e}")

        return chapter_result

    def _create_chapter_translator(self):
        """Create a translator instance for a specific chapter with independent context."""
        # Return the main translator - we'll handle context at the chapter level
        return self.translate_model

    def _translate_with_chapter_context(
        self, translator, text, chapter_context_list, chapter_translated_list
    ):
        """Translate text with chapter-specific context management."""
        if not translator.context_flag:
            return translator.translate(text)

        # Temporarily replace global context with chapter context
        original_context = getattr(translator, "context_list", [])
        original_translated = getattr(translator, "context_translated_list", [])

        try:
            # Use chapter-specific context
            translator.context_list = chapter_context_list.copy()
            translator.context_translated_list = chapter_translated_list.copy()

            # Perform translation
            result = translator.translate(text)

            # Update chapter context
            chapter_context_list[:] = translator.context_list
            chapter_translated_list[:] = translator.context_translated_list

            return result

        finally:
            # Restore original context
            translator.context_list = original_context
            translator.context_translated_list = original_translated

    def _translate_paragraphs_acc_parallel(
        self,
        p_list,
        send_num,
        translator,
        chapter_context_list,
        chapter_translated_list,
    ):
        """Apply accumulated_num logic for a single chapter in parallel mode with independent context."""
        from book_maker.utils import num_tokens_from_text
        from .helper import not_trans

        count = 0
        wait_p_list = []

        # Create chapter-specific helper instance with context-aware translation
        class ChapterHelper:
            def __init__(
                self, parent_loader, translator, context_list, translated_list
            ):
                self.parent_loader = parent_loader
                self.translator = translator
                self.context_list = context_list
                self.translated_list = translated_list

            def translate_with_context(self, text):
                return self.parent_loader._translate_with_chapter_context(
                    self.translator, text, self.context_list, self.translated_list
                )

            def deal_old(self, wait_p_list, single_translate):
                if not wait_p_list:
                    return

                # Use the same translate_list logic as sequential processing
                # Create a temporary translator with chapter context
                original_context = getattr(self.translator, "context_list", [])
                original_translated = getattr(
                    self.translator, "context_translated_list", []
                )

                try:
                    # Set chapter context to the translator
                    self.translator.context_list = self.context_list.copy()
                    self.translator.context_translated_list = (
                        self.translated_list.copy()
                    )

                    # Call translate_list for consistent batch translation logic
                    result_txt_list = self.translator.translate_list(wait_p_list)

                    # Update chapter context from translator
                    self.context_list[:] = self.translator.context_list
                    self.translated_list[:] = self.translator.context_translated_list

                    # Apply translations using the same logic as helper.deal_old
                    for i in range(len(wait_p_list)):
                        if i < len(result_txt_list):
                            p = wait_p_list[i]
                            from .helper import shorter_result_link

                            self.parent_loader.helper.insert_trans(
                                p,
                                shorter_result_link(result_txt_list[i]),
                                self.parent_loader.translation_style,
                                single_translate,
                            )

                finally:
                    # Restore original context
                    self.translator.context_list = original_context
                    self.translator.context_translated_list = original_translated

                wait_p_list.clear()

            def deal_new(self, p, wait_p_list, single_translate):
                self.deal_old(wait_p_list, single_translate)
                translation = self.translate_with_context(p.text)
                self.parent_loader.helper.insert_trans(
                    p,
                    translation,
                    self.parent_loader.translation_style,
                    single_translate,
                )

        chapter_helper = ChapterHelper(
            self, translator, chapter_context_list, chapter_translated_list
        )

        for i in range(len(p_list)):
            p = p_list[i]
            temp_p = copy(p)

            for p_exclude in self._get_exclude_translate_tag_names():
                if type(p) == NavigableString:
                    continue
                for pt in temp_p.find_all(p_exclude):
                    pt.extract()

            if any(
                [not p.text, self._is_special_text(temp_p.text), not_trans(temp_p.text)]
            ):
                if i == len(p_list) - 1:
                    chapter_helper.deal_old(wait_p_list, self.single_translate)
                continue

            length = num_tokens_from_text(temp_p.text)
            if length > send_num:
                chapter_helper.deal_new(p, wait_p_list, self.single_translate)
                continue

            if i == len(p_list) - 1:
                if count + length < send_num:
                    wait_p_list.append(p)
                    chapter_helper.deal_old(wait_p_list, self.single_translate)
                else:
                    chapter_helper.deal_new(p, wait_p_list, self.single_translate)
                break

            if count + length < send_num:
                count += length
                wait_p_list.append(p)
            else:
                chapter_helper.deal_old(wait_p_list, self.single_translate)
                wait_p_list.append(p)
                count = length

    def batch_init_then_wait(self):
        name, _ = os.path.splitext(self.epub_name)
        if self.batch_flag or self.batch_use_flag:
            self.translate_model.batch_init(name)
            if self.batch_use_flag:
                start_time = time.time()
                while not self.translate_model.is_completed_batch():
                    print("Batch translation is not completed yet")
                    time.sleep(2)
                    if time.time() - start_time > 300:  # 5 minutes
                        raise Exception("Batch translation timed out after 5 minutes")

    def _make_bilingual_book_legacy(self):
        new_book = self._make_new_book(self.origin_book)
        all_items = list(self.origin_book.get_items())
        trans_taglist = self._get_translate_tag_names()
        all_p_length = sum(
            (
                0
                if (
                    (i.get_type() != ITEM_DOCUMENT)
                    or (i.file_name in self.exclude_filelist.split(","))
                    or (
                        self.only_filelist
                        and i.file_name not in self.only_filelist.split(",")
                    )
                )
                else len(bs(i.content, "html.parser").find_all(trans_taglist))
            )
            for i in all_items
        )
        all_p_length += self.allow_navigable_strings * sum(
            (
                0
                if (
                    (i.get_type() != ITEM_DOCUMENT)
                    or (i.file_name in self.exclude_filelist.split(","))
                    or (
                        self.only_filelist
                        and i.file_name not in self.only_filelist.split(",")
                    )
                )
                else len(bs(i.content, "html.parser").find_all(string=True))
            )
            for i in all_items
        )
        # Use leave=False in test mode to prevent duplicate progress bar display
        pbar = tqdm(
            total=self.test_num if self.is_test else all_p_length,
            leave=not self.is_test,
        )
        print()
        index = 0
        p_to_save_len = len(self.p_to_save)
        try:
            if self.retranslate:
                self.retranslate_book(
                    index, p_to_save_len, pbar, trans_taglist, self.retranslate
                )
                exit(0)
            # Add the things that don't need to be translated first, so that you can see the img after the interruption
            for item in self.origin_book.get_items():
                if item.get_type() != ITEM_DOCUMENT:
                    new_book.add_item(item)

            document_items = list(self.origin_book.get_items_of_type(ITEM_DOCUMENT))

            if self.enable_parallel and len(document_items) > 1:
                # Optimize worker count: no point having more workers than chapters
                effective_workers = min(self.parallel_workers, len(document_items))

                # Parallel processing with proper accumulated_num handling
                print(f"🚀 Parallel processing: {len(document_items)} chapters")
                if effective_workers < self.parallel_workers:
                    print(
                        f"📊 Optimized workers: {effective_workers} (reduced from {self.parallel_workers})"
                    )
                else:
                    print(f"📊 Using {effective_workers} workers")

                if self.accumulated_num > 1:
                    print(
                        f"📝 Each chapter applies accumulated_num={self.accumulated_num} independently"
                    )

                if self.context_flag:
                    print(
                        f"🔗 Context enabled: each chapter maintains independent context (limit={self.translate_model.context_paragraph_limit})"
                    )
                else:
                    print(f"🚫 Context disabled for this translation")

                # Create a simpler progress bar for parallel processing
                pbar.close()  # Close the original progress bar
                chapter_pbar = tqdm(
                    total=len(document_items), desc="Chapters", unit="ch"
                )

                chapter_data_list = [
                    (item, trans_taglist, p_to_save_len) for item in document_items
                ]

                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    future_to_item = {
                        executor.submit(
                            self._process_chapter_parallel, chapter_data
                        ): chapter_data[0]
                        for chapter_data in chapter_data_list
                    }

                    for future in as_completed(future_to_item):
                        item = future_to_item[future]
                        try:
                            result = future.result()
                            if result["success"] and result["processed_content"]:
                                item.content = result["processed_content"]
                            new_book.add_item(item)
                            chapter_pbar.update(1)
                            chapter_pbar.set_postfix_str(
                                f"Latest: {item.file_name[:20]}..."
                            )

                        except Exception as e:
                            print(f"❌ Error processing {item.file_name}: {e}")
                            new_book.add_item(item)
                            chapter_pbar.update(1)

                chapter_pbar.close()
                print(f"✅ Completed all {len(document_items)} chapters")
            else:
                # Sequential processing (original behavior or single chapter)
                if len(document_items) == 1 and self.enable_parallel:
                    print(f"📄 Single chapter detected - using sequential processing")

                for item in document_items:
                    # Continue processing all chapters (to add them to book)
                    # but skip translation after test limit
                    if self.is_test and index >= self.test_num:
                        # Just add the chapter without translation
                        new_book.add_item(item)
                        continue

                    index = self.process_item(
                        item, index, p_to_save_len, pbar, new_book, trans_taglist
                    )

                # Close progress bar
                pbar.close()

                if self.accumulated_num > 1:
                    name, _ = os.path.splitext(self.epub_name)
                    epub.write_epub(f"{name}_bilingual.epub", new_book, {})
            name, _ = os.path.splitext(self.epub_name)
            if self.batch_flag:
                self.translate_model.batch()
            else:
                epub.write_epub(f"{name}_bilingual.epub", new_book, {})
        except KeyboardInterrupt as e:
            print(e)
            if self.accumulated_num == 1:
                print("you can resume it next time")
                self._save_progress()
                self._save_temp_book()
            sys.exit(0)
        except Exception as e:
            # Handle connection errors gracefully
            error_msg = str(e)
            if "Connection" in error_msg or "connection" in error_msg:
                print(
                    f"[bold red]Translation failed: Connection error - {error_msg}[/bold red]"
                )
                print("Please check your network connection or API server status.")
            else:
                traceback.print_exc()
            if self.accumulated_num == 1:
                print("Saving progress...")
                self._save_progress()
                self._save_temp_book()
            sys.exit(0)

    def make_bilingual_book(self):
        self.helper = EPUBBookLoaderHelper(
            self.translate_model,
            self.accumulated_num,
            self.translation_style,
            self.context_flag,
        )
        self.batch_init_then_wait()

        if self._should_use_checkpoint_pipeline():
            if self.parallel_workers > 1:
                print(
                    "Checkpoint-based EPUB translation does not support parallel chapter processing yet; falling back to sequential mode."
                )
                self.parallel_workers = 1
                self.enable_parallel = False
            self._make_bilingual_book_with_checkpoint()
            return

        if self.resume:
            self.load_state()
        self._make_bilingual_book_legacy()

    def load_state(self):
        if self._should_use_checkpoint_pipeline():
            self._checkpoint_state = self._load_json_checkpoint_state()
            return

        try:
            with open(self.bin_path, "rb") as f:
                self.p_to_save = pickle.load(f)
                self._legacy_state_loaded = True
        except Exception:
            raise Exception("can not load resume file")

    def _save_temp_book_with_checkpoint(self):
        origin_book_temp = epub.read_epub(self.epub_name)
        new_temp_book = self._make_new_book(origin_book_temp)
        try:
            for item in origin_book_temp.get_items():
                if item.get_type() == ITEM_DOCUMENT:
                    context = self._build_document_context(item)
                    if not context["skip"]:
                        for segment in context["segments"]:
                            translated_text = self._checkpoint_state.translations.get(
                                segment.segment_id
                            )
                            if translated_text is not None:
                                self._apply_translation_to_segment(
                                    segment, translated_text
                                )
                        if context["soup"] is not None:
                            item.content = context["soup"].encode(encoding="utf-8")
                new_temp_book.add_item(item)
            name, _ = os.path.splitext(self.epub_name)
            epub.write_epub(f"{name}_bilingual_temp.epub", new_temp_book, {})
        except Exception as e:
            print(e)

    def _save_temp_book_legacy(self):
        origin_book_temp = epub.read_epub(self.epub_name)
        new_temp_book = self._make_new_book(origin_book_temp)
        p_to_save_len = len(self.p_to_save)
        trans_taglist = self._get_translate_tag_names()
        index = 0
        try:
            for item in origin_book_temp.get_items():
                if item.get_type() == ITEM_DOCUMENT:
                    content = item.content
                    soup = bs(content, "html.parser")
                    p_list = soup.find_all(trans_taglist)
                    if self.allow_navigable_strings:
                        p_list.extend(soup.find_all(string=True))
                    for p in p_list:
                        if not p.text or self._is_special_text(p.text):
                            continue
                        if index < p_to_save_len:
                            new_p = copy(p)
                            if type(p) is NavigableString:
                                new_p = self.p_to_save[index]
                            else:
                                new_p.string = self.p_to_save[index]
                            self.helper.insert_trans(
                                p,
                                new_p.string,
                                self.translation_style,
                                self.single_translate,
                            )
                            index += 1
                        else:
                            break
                    if soup:
                        item.content = soup.encode()
                new_temp_book.add_item(item)
            name, _ = os.path.splitext(self.epub_name)
            epub.write_epub(f"{name}_bilingual_temp.epub", new_temp_book, {})
        except Exception as e:
            print(e)

    def _save_temp_book(self):
        if self._should_use_checkpoint_pipeline():
            if self._checkpoint_state is None:
                self._checkpoint_state = self._default_checkpoint_state()
            self._save_temp_book_with_checkpoint()
            return

        self._save_temp_book_legacy()

    def _save_progress(self):
        if self._should_use_checkpoint_pipeline():
            self._save_json_checkpoint()
            return

        try:
            with open(self.bin_path, "wb") as f:
                pickle.dump(self.p_to_save, f)
        except Exception:
            raise Exception("can not save resume file")
