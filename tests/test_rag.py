"""Открытые тесты для студентов."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Для тестов нужны реальные ключи
ROOT = Path(__file__).resolve().parents[1]

try:
    import sys
    sys.path.insert(0, str(ROOT))
    from rag_pipeline import GOLDEN_QA, ask
except ImportError as e:
    pytest.skip(f'Нет модуля rag_pipeline: {e}', allow_module_level=True)


def test_p1_webvisor():
    """P1: ask('Что такое Вебвизор?') возвращает корректный ответ."""
    out = ask('Что такое Вебвизор?')
    
    # Структура ответа
    assert set(out) == {'answer', 'sources', 'cost_rub'}, 'Ответ должен содержать ровно три ключа'
    
    # answer
    assert isinstance(out['answer'], str), 'answer должен быть строкой'
    assert len(out['answer'].strip()) >= 10, 'answer слишком короткий'
    
    # sources
    assert isinstance(out['sources'], list), 'sources должен быть списком'
    assert 1 <= len(out['sources']) <= 5, 'sources должен содержать от 1 до 5 путей'
    assert all(isinstance(p, str) for p in out['sources']), 'Все sources должны быть строками'
    assert any('webvisor' in p for p in out['sources']), 'В sources должен быть путь с webvisor'
    
    # cost_rub
    assert isinstance(out['cost_rub'], (int, float)), 'cost_rub должен быть числом'
    assert out['cost_rub'] > 0, 'cost_rub должен быть положительным'


def test_p2_install():
    """P2: ask('Куда поставить код счётчика?') возвращает релевантные пути."""
    out = ask('Куда поставить код счётчика Яндекс Метрики?')
    
    # Структура ответа
    assert set(out) == {'answer', 'sources', 'cost_rub'}
    
    # Проверяем, что хотя бы один из путей релевантен
    paths = ' '.join(out['sources'])
    assert any(
        kw in paths for kw in ('install', 'creating-counter', 'quick-start', 'check-counter')
    ), f'В sources нет релевантных путей для установки счётчика: {out["sources"]}'
    
    assert out['cost_rub'] > 0


def _p_at_k(sources: list[str], relevant: set[str], k: int = 3) -> float:
    """Precision@k с поддержкой префиксов."""
    top = sources[:k]
    if not top:
        return 0.0
    
    def _hit(path: str) -> bool:
        return any(
            path == r or path.startswith(r.rstrip('/') + '/') or path.startswith(r) or r in path
            for r in relevant
        )
    
    return sum(1 for p in top if _hit(p)) / min(k, len(top))


def test_p3_golden_qa():
    """P3: ask проходит все 5 вопросов из GOLDEN_QA с mean precision@3 >= 0.4."""
    scores = []
    
    for case in GOLDEN_QA:
        out = ask(case['question'])
        
        # Базовые проверки
        assert set(out) == {'answer', 'sources', 'cost_rub'}
        assert len(out['answer'].strip()) >= 10, f'Ответ слишком короткий для {case["id"]}'
        assert out['cost_rub'] > 0, f'cost_rub должен быть > 0 для {case["id"]}'
        
        # Precision@3
        relevant = set(case['relevant_paths'])
        p = _p_at_k(out['sources'], relevant, k=3)
        scores.append(p)
    
    mean_p = sum(scores) / len(scores) if scores else 0
    assert mean_p >= 0.4, f'mean precision@3 слишком низкая: {mean_p:.3f}, scores={scores}'
