from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    text: str
    section_title: str | None
    token_count: int


@dataclass(frozen=True)
class TextBlock:
    text: str
    section_title: str | None


def approx_token_count(text: str) -> int:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?", text))
    punctuation = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))
    return max(1, cjk_chars + latin_words + punctuation)


def chunk_text(
    text: str,
    *,
    target_tokens: int = 600,
    overlap_tokens: int = 80,
) -> list[ChunkDraft]:
    blocks = _parse_blocks(text)
    chunks: list[ChunkDraft] = []
    current_parts: list[str] = []
    current_section: str | None = None

    def emit(*, keep_overlap: bool = True) -> None:
        nonlocal current_parts, current_section
        merged = "\n\n".join(part.strip() for part in current_parts if part.strip()).strip()
        if not merged:
            current_parts = []
            return
        chunks.append(
            ChunkDraft(
                chunk_index=len(chunks),
                text=merged,
                section_title=current_section,
                token_count=approx_token_count(merged),
            )
        )
        overlap = _tail_by_approx_tokens(merged, overlap_tokens) if keep_overlap else ""
        current_parts = [overlap] if overlap else []

    for block in blocks:
        split_blocks = _split_oversized_block(block, target_tokens)
        for piece in split_blocks:
            piece_tokens = approx_token_count(piece.text)
            current_tokens = approx_token_count("\n\n".join(current_parts)) if current_parts else 0
            section_changed = current_section is not None and piece.section_title != current_section

            if current_parts and (section_changed or current_tokens + piece_tokens > target_tokens):
                emit(keep_overlap=not section_changed)

            if current_section is None or not current_parts:
                current_section = piece.section_title

            current_parts.append(piece.text)

    if current_parts:
        merged = "\n\n".join(part.strip() for part in current_parts if part.strip()).strip()
        if merged:
            chunks.append(
                ChunkDraft(
                    chunk_index=len(chunks),
                    text=merged,
                    section_title=current_section,
                    token_count=approx_token_count(merged),
                )
            )

    return chunks


def _parse_blocks(text: str) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    current_section: str | None = None
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        value = "\n".join(paragraph).strip()
        if value:
            blocks.append(TextBlock(text=value, section_title=current_section))
        paragraph = []

    for line in text.splitlines():
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading:
            flush()
            current_section = heading.group(1).strip()
            continue
        if not line.strip():
            flush()
            continue
        paragraph.append(line.rstrip())

    flush()
    return blocks


def _split_oversized_block(block: TextBlock, target_tokens: int) -> list[TextBlock]:
    if approx_token_count(block.text) <= target_tokens:
        return [block]

    sentences = re.split(r"(?<=[。！？.!?])\s*", block.text)
    pieces: list[TextBlock] = []
    current: list[str] = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        candidate = "".join(current + [sentence]).strip()
        if current and approx_token_count(candidate) > target_tokens:
            pieces.append(TextBlock(text="".join(current).strip(), section_title=block.section_title))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        pieces.append(TextBlock(text="".join(current).strip(), section_title=block.section_title))

    return pieces


def _tail_by_approx_tokens(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    pieces = re.findall(
        r"\s+|[\u4e00-\u9fff]|[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?|[^\w\s]",
        text,
    )
    if not pieces:
        return ""
    tail: list[str] = []
    used_tokens = 0
    for piece in reversed(pieces):
        piece_tokens = 0 if piece.isspace() else approx_token_count(piece)
        if piece_tokens and used_tokens + piece_tokens > token_budget:
            break
        tail.append(piece)
        used_tokens += piece_tokens
    return "".join(reversed(tail)).strip()
