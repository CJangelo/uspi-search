from dataclasses import dataclass


@dataclass
class LabelRecord:
    label_id: str
    source: str                      # 'openfda_json' | 'dailymed_xml'
    metadata: dict
    sections: list[tuple[str, str]]  # [(section_name, text), ...]
