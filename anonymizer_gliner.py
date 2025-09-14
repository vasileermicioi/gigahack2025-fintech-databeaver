"""
Hackathon Anonymizer 

The evaluator expects anonymize(text) -> (anonymized_text, metadata) where
metadata["entities"] is a list of dicts with:
  { start, end, label, text, replacement }
so that deanonymize(anonymized_text, metadata) restores the original text.
"""
from typing import Tuple, Dict, List, Any
from gliner import GLiNER
from labels.en import DESCRIPTION_LABEL_MAP


class Anonymizer:
    """Minimal anonymizer that you can use as a starting point.
    Only change MODEL_NAME and (optionally) LABEL_MAP above.
    """

    def __init__(self, model_path = "urchade/gliner_multi_pii-v1"):
        self.model = GLiNER.from_pretrained(model_path)

    def anonymize(self, text: str) -> Tuple[str, Dict]:
        labels = list(DESCRIPTION_LABEL_MAP.keys())
        preds = self.model.predict_entities(text, labels, threshold=0.5)
        predicted_spans: List[Dict[str, Any]] = []
        for p in preds:
            s = int(p.get("start", 0))
            e = int(p.get("end", 0))
            word = p.get("text", "")
            label = DESCRIPTION_LABEL_MAP.get(p.get("label", ""), "")
            if e > s:
                predicted_spans.append({
                    "start": s,
                    "end": e,
                    "label": label,
                    "text": word if word else text[s:e],
                })

        predicted_spans.sort(key=lambda s: s["start"])  # left-to-right

        # Build anonymized text with placeholders and metadata
        parts: List[str] = []
        cursor = 0
        entities_meta: List[Dict[str, Any]] = []
        for idx, span in enumerate(predicted_spans, start=1):
            s, e, label = span["start"], span["end"], span["label"]
            placeholder = f"<{label}_{idx}>"
            parts.append(text[cursor:s])
            parts.append(placeholder)
            cursor = e
            entities_meta.append({
                "start": s,
                "end": e,
                "label": label,
                "text": text[s:e],
                "replacement": placeholder,
            })
        parts.append(text[cursor:])
        anon_text = "".join(parts)
        metadata = {"entities": entities_meta}
        return anon_text, metadata

    def deanonymize(self, text: str, metadata: Dict) -> str:
        entities = metadata.get("entities", []) if isinstance(metadata, dict) else []
        result = text
        for ent in reversed(entities):
            result = result.replace(ent["replacement"], ent["text"], 1)
        return result
