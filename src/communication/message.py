from dataclasses import dataclass, asdict

@dataclass
class Message:
    msg_id: str
    src: str
    dst: str
    generated_step: int
    deliver_step: int | None
    payload: dict
    priority: int
    dropped: bool
    sequence_id: int

    def to_dict(self):
        return asdict(self)