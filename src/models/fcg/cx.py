import pprint
import re


class Construction:
    def __init__(self, form: list[str] = None, meaning: list[str] = None, tag: str = None):
        self.form_segments: list[str] = [] if form is None else form
        self.meaning_segments: list[str] = [] if meaning is None else meaning
        self.tag = 'None' if tag is None else tag

    @property
    def form(self):
        return ' '.join(self.form_segments)

    @property
    def meaning(self):
        return ' '.join(self.meaning_segments)

    def __eq__(self, other: 'Construction'):
        return (self.tag == other.tag
                and tuple(self.form_segments) == tuple(other.form_segments)
                and tuple(self.meaning_segments) == tuple(other.meaning_segments))

    def __str__(self):
        return pprint.pformat(repr(self), width=80, sort_dicts=False)

    def __repr__(self):
        return repr(dict(zip(('form', 'meaning', 'tag'), self.serialize())))

    def serialize(self) -> tuple[str, str, str]:
        return (' '.join(self.form_segments),
                ' '.join(self.meaning_segments),
                self.tag)

    @classmethod
    def deserialize(cls, form_str: str, meaning_str: str, tag: str) -> 'Construction':
        slot_pat = re.compile(r'<\d+>')
        assert len(slot_pat.findall(form_str)) == len(slot_pat.findall(meaning_str)), 'slots must be in pairs'

        cx = cls(tag=tag)

        def _add_pieces_of(text: str, to: list[str]):
            start = 0
            for i, m in enumerate(re.finditer(r'<\d+>', text)):
                cur = m.start()
                if start < cur:
                    to.append(text[start:cur].strip())
                to.append(f'<{i}>')
                start = m.end()
                if start == len(text):
                    break
            else:
                to.append(text[start:].strip())

        _add_pieces_of(form_str, to=cx.form_segments)
        _add_pieces_of(meaning_str, to=cx.meaning_segments)
        return cx
