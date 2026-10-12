"""Шаблон для реализации RAG по справке Яндекс Метрики."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import rank_bm25
from openai import OpenAI

# Цена в курсе: 0,1 ₽ за 1000 токенов (вход + выход).
PRICE_PER_1K_RUB = 0.1

# Путь до выгрузки документации Яндекс Метрики
DEFAULT_DOCS_DIR = Path('metrica-docs-snapshot/docs')

# Служебные слова для очистки запроса перед BM25.
QUERY_STOPWORDS = frozenset(
    {
        'и',
        'в',
        'на',
        'по',
        'для',
        'а',
        'с',
        'к',
        'из',
        'не',
        'у',
        'или',
        'это',
        'что',
        'как',
        'где',
        'такое',
    }
)

# Тестовые вопросы
GOLDEN_QA = [
    {
        'id': 'q1',
        'question': 'Куда поставить код счётчика Яндекс Метрики?',
        'relevant_paths': [
            'ru/troubleshooting/install-tag.md',
            'ru/general/creating-counter.md',
            'ru/quick-start.md',
            'ru/code/install',
        ],
    },
    {
        'id': 'q2',
        'question': 'Как проверить, что счётчик установлен корректно?',
        'relevant_paths': [
            'ru/general/check-counter.md',
            'ru/troubleshooting/install-tag.md',
            'ru/quick-start.md',
        ],
    },
    {
        'id': 'q3',
        'question': 'Что такое Вебвизор?',
        'relevant_paths': ['ru/webvisor', 'ru/general/counter-webvisor.md'],
    },
    {
        'id': 'q4',
        'question': 'Как создать цель в Метрике?',
        'relevant_paths': [
            'ru/general/goal',
            'ru/general/choose-goal.md',
            'ru/simple-goal',
            'ru/integrated-goal',
            'ru/general/goals.md',
        ],
    },
    {
        'id': 'q5',
        'question': 'Почему в отчётах нет данных после установки счётчика?',
        'relevant_paths': [
            'ru/troubleshooting',
            'ru/general/check-counter.md',
            'ru/general/sources-qanda.md',
        ],
    },
]

SYSTEM_PROMPT = """<SYSTEM_INSTRUCTION>
Отвечай на вопрос только на основе данных из блока CONTEXT.
Если в контексте нет нужных данных, честно сообщи об этом и ничего не выдумывай.
В ответе явно указывай path использованных источников.
</SYSTEM_INSTRUCTION>
"""


PROMPT = """<CONTEXT>
{context}
</CONTEXT>

<QUESTION>
{question}
</QUESTION>"""


@dataclass
class Document:
    name: str
    relative_path: str
    chunks: list[str]


@dataclass(frozen=True)
class RetrievedChunk:
    """Фрагмент, отобранный retrieval-этапом для передачи в prompt."""

    path: str
    title: str
    text: str
    score: float


TOKEN_RE = re.compile(r'[A-Za-zА-Яа-я0-9_]+')


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def query_tokens(question: str) -> list[str]:
    tokens = tokenize(question)
    without_stopwords = [token for token in tokens if token not in QUERY_STOPWORDS]
    return without_stopwords or tokens


@dataclass
class SearchIndex:
    corpus: list[tuple[Document, str]]
    metadata_tokens: list[set[str]]
    bm25: rank_bm25.BM25Okapi | None


def build_search_index(documents: list[Document]) -> SearchIndex:
    corpus = [
        (document, chunk)
        for document in documents
        for chunk in document.chunks
    ]
    tokenized = [tokenize(chunk) for _, chunk in corpus]

    bm25 = rank_bm25.BM25Okapi(tokenized) if any(tokenized) else None
    metadata_tokens = [
        set(tokenize(f'{document.relative_path} {document.name}'))
        for document, _ in corpus
    ]
    return SearchIndex(corpus, metadata_tokens, bm25)


@lru_cache(maxsize=1)
def get_search_index(docs_dir: Path) -> SearchIndex:
    return build_search_index(indexation(docs_dir))


def retrieve_top_chunks(
    question: str,
    index: SearchIndex,
    top_k: int = 3,
    metadata_bonus: float = 2.0,
) -> list[RetrievedChunk]:
    if top_k <= 0:
        return []

    if not index.corpus:
        return []

    tokens = query_tokens(question)
    bm25_scores = (
        index.bm25.get_scores(tokens)
        if index.bm25 is not None else [0.0] * len(index.corpus)
    )
    unique_query_tokens = set(tokens)

    candidates: list[RetrievedChunk] = []
    for (document, chunk), metadata_tokens, bm25_score in zip(
        index.corpus, index.metadata_tokens, bm25_scores
    ):
        path = str(document.relative_path)
        bonus = metadata_bonus * len(unique_query_tokens & metadata_tokens)
        candidates.append(
            RetrievedChunk(
                path=path,
                title=document.name,
                text=chunk,
                score=float(bm25_score + bonus),
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)

    selected: list[RetrievedChunk] = []
    selected_paths: set[str] = set()
    for candidate in candidates:
        if candidate.path in selected_paths:
            continue
        selected.append(candidate)
        selected_paths.add(candidate.path)
        if len(selected) == top_k:
            break

    return selected


def split_into_chunks(text: str, chunk_size: int = 4000) -> list[str]:
    if chunk_size <= 0:
        raise ValueError('chunk_size должен быть положительным')
    if not text.strip():
        return []

    # Сохраняем исходные разделители и отступы, включая блоки кода Markdown.
    parts = re.split(r'(\r?\n[ \t]*\r?\n)', text)
    paragraphs = [
        parts[i] + (parts[i + 1] if i + 1 < len(parts) else '')
        for i in range(0, len(parts), 2)
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if not paragraph:
            continue

        # Если абзац больше лимита, режем по длине
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            for i in range(0, len(paragraph), chunk_size):
                part = paragraph[i:i + chunk_size]

                if len(part) == chunk_size:
                    chunks.append(part)
                else:
                    current_chunk = part

            continue

        # Проверяем, помещается ли абзац в текущий чанк
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def indexation(docs_dir: Path = DEFAULT_DOCS_DIR) -> list[Document]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f'Каталог документации не найден: {docs_dir}')
    documents = []
    for file in sorted(docs_dir.rglob('*.md')):
        content = file.read_text(encoding='utf-8')
        title = next(
            (line.removeprefix('# ').strip()
             for line in content.splitlines() if line.startswith('# ') and line[2:].strip()),
            file.stem,
        )
        relative_path = file.relative_to(docs_dir).as_posix()
        documents.append(
            Document(
                name=title,
                relative_path=relative_path,
                chunks=split_into_chunks(content)
                )
            )

    return documents


def augmentation(question: str, retrieve_chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f'[{number}]\npath={chunk.path}\n{chunk.text}'
        for number, chunk in enumerate(retrieve_chunks, start=1)
    )
    return PROMPT.format(context=context, question=question)


def llm_call(prompt: str) -> dict:
    api_key = os.getenv('LLM_API_KEY', '')
    base_url = os.getenv('LLM_BASE_URL', 'https://llm.api.cloud.yandex.net/v1')
    model = os.getenv('LLM_MODEL', '')

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0,
        max_tokens=1500,
    )

    answer = response.choices[0].message.content or ''

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    cost_rub = round(
        (prompt_tokens + completion_tokens) / 1000 * PRICE_PER_1K_RUB,
        4
    )

    return {
        'answer': answer,
        'cost_rub': cost_rub,
    }


def ask(question: str) -> dict:
    """
    Отвечает на вопрос по документации Яндекс Метрики.
    
    Возвращает словарь с ключами:
    - answer: str — текст ответа модели
    - sources: list[str] — список путей файлов источников (до 3)
    - cost_rub: float — стоимость запроса в рублях
    """
    index = get_search_index(DEFAULT_DOCS_DIR.resolve())
    chunks = retrieve_top_chunks(question, index)
    prompt = augmentation(question, chunks)
    result = llm_call(prompt)
    return {
        'answer': result['answer'],
        'sources': [chunk.path for chunk in chunks],
        'cost_rub': result['cost_rub'],
    }


if __name__ == '__main__':
    print(ask('Отчет география'))
