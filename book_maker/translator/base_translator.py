import itertools
import json
import re
from abc import ABC, abstractmethod

# Special delimiter for batch translation - UUID-based token unlikely to appear in any text
BATCH_DELIMITER = "\n\n@@\n\n"


class Base(ABC):
    def __init__(self, key, language) -> None:
        self.keys = itertools.cycle(key.split(","))
        self.language = language

    @abstractmethod
    def rotate_key(self):
        pass

    @abstractmethod
    def translate(self, text):
        pass

    def translate_segments(self, segments):
        translated_segments = []
        for segment in segments:
            translated_segments.append(
                {
                    "id": segment["id"],
                    "translation": self.translate(segment["text"]),
                }
            )
        return translated_segments

    def build_segment_translation_prompt(self, segments, prompt_template=""):
        try:
            instruction_template = prompt_template.format(
                text="__SEGMENT_TEXT__",
                language=self.language,
                crlf="\n",
            )
        except Exception:
            instruction_template = prompt_template or ""

        payload = json.dumps(segments, ensure_ascii=False, indent=2)
        return (
            f"Translate every segment to {self.language}.\n"
            "Return only valid JSON.\n"
            'The output must be a JSON array of objects with exactly two keys: "id" and "translation".\n'
            "Preserve every segment id exactly.\n"
            "Do not omit, duplicate, merge, or split segments.\n"
            "The number of output objects must equal the number of input objects.\n"
            "Each translation must correspond to the segment with the same id.\n"
            "If a translation is intentionally empty, return an empty string.\n"
            "Do not wrap the JSON in Markdown explanations.\n"
            "Apply the same translation intent as this per-segment instruction template:\n"
            f"{instruction_template}\n\n"
            "Input JSON:\n"
            f"{payload}"
        )

    def parse_segment_translation_response(self, content):
        if content is None:
            raise ValueError("segment translation response is empty")

        candidate = content.strip()
        fenced_match = re.search(
            r"```(?:json)?\s*(\[.*\])\s*```", candidate, re.DOTALL | re.IGNORECASE
        )
        if fenced_match:
            candidate = fenced_match.group(1).strip()
        else:
            start = candidate.find("[")
            end = candidate.rfind("]")
            if start != -1 and end != -1 and end >= start:
                candidate = candidate[start : end + 1]

        parsed = json.loads(candidate)
        if not isinstance(parsed, list):
            raise ValueError("segment translation response must be a JSON array")
        return parsed

    def set_deployment_id(self, deployment_id):
        pass

    def translate_list(self, text_list):
        """
        Translate a list of texts. Default implementation translates one by one.
        Subclasses can override for batch efficiency.
        """
        return [self.translate(t) for t in text_list]
