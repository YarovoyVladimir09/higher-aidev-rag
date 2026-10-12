"""Шаблон для реализации RAG по справке Яндекс Метрики."""

from __future__ import annotations

from dataclasses import dataclass

import os

import rank_bm25

import re

from pathlib import Path

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


@dataclass
class Document:
    name: str
    relative_path: str
    content: str
    chunks: list[str]


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def split_into_chunks(text: str, chunk_size: int = 4000) -> list[str]:
    # Разбиваем текст по пустым строкам
    paragraphs = re.split(r"\n\s*\n", text.strip())

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

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
        separator = "\n\n" if current_chunk else ""

        if len(current_chunk) + len(separator) + len(paragraph) <= chunk_size:
            current_chunk += separator + paragraph
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def indexation(docs_dir: Path = DEFAULT_DOCS_DIR) -> list[Document]:
    documents = []
    for file in docs_dir.rglob('*.md'):

        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("# "):
                    continue

                title = line
                break
            else:
                print(f"No title found in {file}")
        
        title = normalize(title.removeprefix("# ").strip())

        content = normalize(file.read_text(encoding="utf-8"))
        relative_path = file.relative_to(docs_dir)
        # print(f"Path: {relative_path}")

        documents.append(
            Document(
                name=title,
                relative_path=relative_path,
                content=content,
                chunks=split_into_chunks(content)
                )
            )

    return documents


def ask(question: str) -> dict:
    """
    Отвечает на вопрос по документации Яндекс Метрики.
    
    Возвращает словарь с ключами:
    - answer: str — текст ответа модели
    - sources: list[str] — список путей файлов источников (до 3)
    - cost_rub: float — стоимость запроса в рублях
    """
    # TODO: Реализуйте RAG pipeline:
    # 1. Indexing: прочитать и разбить документы на фрагменты
    # 2. Retrieval: найти top-3 релевантных фрагмента (BM25 + бонус за метаданные)
    # 3. Augmentation: собрать промпт с контекстом
    # 4. Generation: вызвать модель и вернуть результат
    raise NotImplementedError('Реализуйте функцию ask')


indx = indexation()
print("\n".join(sorted(doc.name for doc in indx)))

