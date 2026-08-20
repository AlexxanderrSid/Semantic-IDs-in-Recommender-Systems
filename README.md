# Семантические ID в рекомендательных системах

Выпускная квалификационная работа (бакалавриат), Факультет компьютерных наук НИУ ВШЭ, образовательная программа «Прикладная математика и информатика».

**Автор:** Сидоров Александр Михайлович, БПМИ223
**Москва, 2026**

Полный текст работы: [`report.pdf`](./report.pdf)

## О работе

Генеративные рекомендательные системы — молодое направление, в котором рекомендации для пользователя генерируются авторегрессивно, токен за токеном, по аналогии с генерацией текста в языковых моделях. Центральное понятие этого подхода — **семантические идентификаторы (Semantic IDs, S-IDs)**: дискретные иерархические коды айтемов, которые, в отличие от произвольных числовых индексов, кодируют семантическую близость между объектами.

В работе проведено систематическое исследование свойств семантических идентификаторов:

1. предложена и обоснована система из шести **intrinsic-метрик** качества S-IDs (Collision Rate, Codebook Utilization, Semantic Similarity Preservation, Token Predictability Score, Nearest Neighbour Alignment, Silhouette Score);
2. проведено сравнение методов построения S-IDs — **RKMeans**, **RVQ** и **RQ-VAE**;
3. исследовано влияние ключевых гиперпараметров кодового дерева — глубины `L` и ширины кодбука `W`;
4. проверена гипотеза об обогащении S-IDs поведенческим сигналом (кодирование периодичности взаимодействий пользователя с айтемом);
5. выполнен корреляционный анализ intrinsic- и extrinsic-метрик для поиска дешёвого прокси-критерия качества рекомендаций.

Эксперименты проводились на датасете **Amazon Reviews Beauty** (12 101 товар после предобработки), эмбеддинги текстовых описаний товаров получены с помощью **Flan-T5-Large**, рекомендательная модель — **TIGER** (архитектура T5 encoder-decoder) в реализации фреймворка **GRID**.

## Ключевые результаты

| Эксперимент | Вывод |
|---|---|
| Сравнение методов квантования | **RKMeans** превосходит RVQ и RQ-VAE по всем extrinsic-метрикам, несмотря на то что RQ-VAE даёт лучший CR и SSP — сложность метода квантования не гарантирует лучшего качества рекомендаций |
| Глубина кодового дерева `L` | Оптимум при `L = 3`: `L = 2` создаёт слишком много коллизий, `L = 4` затрудняет обучение генеративной модели при фиксированном бюджете шагов |
| Ширина кодбука `W` | Компактный кодбук (`W = 128`) даёт лучший Recall@10, чем `W = 256` и `W = 512`, за счёт более плотной и обучаемой структуры кодового пространства |
| Кодирование периодичности | Ни одна из трёх стратегий (`interval`, `topk_pos`, `binary_bag`) не улучшила качество рекомендаций — прямое подмешивание поведенческого сигнала в эмбеддинг перед квантованием конфликтует с семантической структурой |
| Корреляция intrinsic/extrinsic | **SSP (Semantic Similarity Preservation)** — лучший предиктор extrinsic-качества среди всех intrinsic-метрик (средний |r| = 0.546 по Пирсону) и может использоваться как дешёвый прокси-критерий при отборе конфигурации без полного обучения модели |

Подробные таблицы метрик и обсуждение — в разделе 5 отчёта.

## Структура репозитория

```
.
├── report.pdf          # текст ВКР
└── GRID/                # фреймворк генеративных рекомендаций с семантическими ID,
    │                     # использованный как база для экспериментов
    ├── configs/          # конфигурации Hydra (обучение/инференс эмбеддингов, S-ID, TIGER)
    ├── src/               # исходный код: квантование, датамодули, модель TIGER, метрики
    ├── requirements.txt
    └── README.md         # документация фреймворка GRID (Snap Research)
```

`GRID` — фреймворк [snap-research/GRID](https://github.com/snap-research/GRID) для генеративных рекомендаций на основе семантических ID; в этой работе он используется как экспериментальный стенд для обучения квантизаторов (RKMeans / RVQ / RQ-VAE) и модели TIGER.

## Воспроизведение экспериментов

```bash
cd GRID
pip install -r requirements.txt
```

**1. Генерация эмбеддингов товаров** (Flan-T5-Large поверх текстовых описаний):

```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty
```

**2. Обучение квантизатора и построение семантических ID** (пример для RKMeans, `L=3`, `W=256` — базовая конфигурация A1 из отчёта):

```bash
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<путь_из_шага_1>/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256

python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<путь_из_шага_1>/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path=<чекпоинт_из_шага_2>
```

Для сравнения методов используйте `rvq_train_flat` / `rvq_inference_flat` и `rqvae_train_flat` / `rqvae_inference_flat`; для ablation по глубине и ширине кодового дерева — варьируйте `num_hierarchies` и `codebook_width`.

**3. Обучение и инференс генеративной рекомендательной модели TIGER:**

```bash
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=<путь_из_шага_2>/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4  # +1 к num_hierarchies из шага 2 — дополнительный разряд для дедупликации коллизий

python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=<путь_из_шага_2>/pickle/merged_predictions_tensor.pt \
    ckpt_path=<чекпоинт_из_шага_3> \
    num_hierarchies=4
```

Подробнее о конфигурациях, параметрах и предобработанных данных Amazon — в [`GRID/README.md`](./GRID/README.md).

## Основные источники

- Rajput, S. et al. *Recommender systems with generative retrieval.* NeurIPS 2023. (TIGER)
- Deng, J. et al. *OneRec: Unifying retrieve and rank with generative recommender and iterative preference alignment.* arXiv:2502.18965. (Residual K-means)
- Lee, D. et al. *Autoregressive image generation using residual quantization.* CVPR 2022. (RQ-VAE)
- Ju, C. M. et al. *Generative Recommendation with Semantic IDs: A Practitioner's Handbook.* CIKM 2025. (GRID)

Полный список литературы — в отчёте, раздел «Список литературы».

## Лицензия

Код фреймворка `GRID` распространяется на условиях, указанных в [`GRID/LICENSE`](./GRID/LICENSE).
