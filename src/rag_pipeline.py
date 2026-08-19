"""Шаблон для реализации RAG по справке Яндекс Метрики."""

from __future__ import annotations

import os
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
