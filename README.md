# LLM-classifier

LLM-классификатор опасности configuration drift в Kubernetes. Каждому drift-событию (расхождение между `desired` и `live` манифестом) модель присваивает класс опасности от 1 до 4 по той же инструкции разметки, по которой кейсы размечали эксперты.

| Класс | Смысл |
|---|---|
| 1 | Низкий: поведение, доступность и безопасность не меняются |
| 2 | Средний: меняется поведение или конфигурация, прямой угрозы нет |
| 3 | Высокий: возможна недоступность сервиса или ослаблена защита |
| 4 | Критический: прямой путь к компрометации узла, кластера или учётных данных |

## Как это работает

- **Вход модели.** На каждый кейс модель получает вид, имя, namespace и окружение (`dev`/`prod`) ресурса, человекочитаемый diff (`diff_readable`) и `desired.yaml`. Метаданные генератора корпуса (операторы мутаций, тип кейса и т. п.) во вход не попадают.
- **Промпт.** Системная часть — роль, задача и разделы 1, 3, 4, 6 [инструкции разметки](https://github.com/VityazevEgor/DriftBench-Offline-Pub/blob/main/dataset/annotation/Rubric_DriftBench_Annotation.md) корпуса. Пользовательское сообщение строится по шаблону `prompts/user_template_v2.md`; `desired.yaml` длиннее 20 000 символов обрезается. Готовый системный промпт — `prompts/prompt_v2.md`.
- **Ответ.** Строго JSON `{"label": 1..4, "explanation": "..."}`. Невалидный ответ запрашивается заново (до 2 повторов), после этого кейс считается ошибкой классификации (`label = 0`).
- **Параметры вызова.** `temperature = 0`, `max_tokens = 4000` (при исчерпании лимита рассуждением повтор с 8000), до 4 параллельных запросов, до 5 попыток на 429/5xx.
- **Итоговое предсказание.** Каждый кейс классифицируется в 3 независимых прогонах; итог — мода (если все три ответа различны — ответ первого прогона).
- **Модели.** DeepSeek v4.1 Flash и Qwen3.6-35B-A3B через OpenAI-совместимый `chat/completions` Yandex Cloud AI Studio.

## Структура

```
llm_classifier/   клиент API, сборка сообщений, разбор ответа, голосование по прогонам
prompts/          части промпта, шаблон сообщения, build_prompt.py и собранные промпты
prepare_data.py   сборка входов модели и эталона по сплитам
run.py            прогон модели
evaluate.py       метрики
tests/            pytest-тесты (без обращения к API)
data/             входы моделей, эталонные метки по сплитам, manifest.json
results/          логи запросов, метрики, итоговые предсказания, SHA-256 промпта
config.yaml       пути, параметры вызова, URI моделей
```

## Запуск

Корпус DriftBench-Offline (кейсы, разбиение на dev/test, эталонная разметка, инструкция разметки) опубликован в репозитории [DriftBench-Offline-Pub](https://github.com/VityazevEgor/DriftBench-Offline-Pub/tree/main). Клон должен лежать рядом с этим репозиторием (пути настраиваются в `config.yaml`):

```
<workspace>/
├── DriftBench-Offline-Pub/
└── LLM-classifier-pub/
```

```bash
git clone https://github.com/VityazevEgor/DriftBench-Offline-Pub.git
pip install -r requirements.txt
cp .env.example .env    # впишите ключ API и идентификатор каталога Yandex Cloud
```

Ключ хранится только в `.env`; путь к файлу можно переопределить переменной окружения `YANDEX_ENV_FILE`.

```bash
python prepare_data.py                                    # data/input, data/labels_*.csv, data/manifest.json
python prompts/build_prompt.py 2                          # пересобрать prompts/prompt_v2.md из инструкции разметки
python run.py --model deepseek --split test --runs 3 --prompt-version 2
python run.py --model qwen --split test --runs 3 --prompt-version 2
python evaluate.py --model deepseek --split test
python evaluate.py --model qwen --split test
python -m pytest tests
```

Быстрая проверка работоспособности: один запрос по одному кейсу с выводом предсказания и эталона в консоль, без записи файлов (по умолчанию — первый по номеру кейс):

```bash
python run.py --model deepseek --check            # или --check CASE_0030
```

Прогон возобновляемый: ответы, уже записанные в `results/raw_*.jsonl`, повторно не запрашиваются. Готовые входы лежат в `data/`, готовые промпты — в `prompts/`, поэтому `prepare_data.py` и `build_prompt.py` нужны только для пересборки. Folder ID Yandex Cloud в опубликованных логах заменён на `<FOLDER_ID>`.

## Результаты

Подробные метрики: `results/metrics_<model>_test.json`. Итоговые предсказания: `results/predictions_<model>_test.csv`. SHA-256 промпта: `results/prompt_frozen.sha256`.
