# NLP Code Generation

Обучение encoder–decoder Transformer для генерации кода по текстовому вопросу: подготовка корпуса, обычный CE, дополнительные задачи, обработка градиентов, возобновление обучения и inference. В репозитории находятся исходники и конфигурации; данные и веса подключаются отдельно. По умолчанию модель инициализируется случайно.

## Установка

Запускай команды из корня проекта с Python 3.10+:

```bash
python -m pip install -e .
```

Установленный PyTorch должен поддерживать нужное устройство. Для CPU/MPS используй `train.precision=fp32`; стандартная конфигурация обучения рассчитана на CUDA с FP16.

## Данные

`data.mode=prepared` используется по умолчанию: сначала подготовь корпус командой ниже. В `data.mode=online` исходные строки токенизируются на CPU во время обучения; предварительно токенизировать весь корпус не требуется.

### Подготовленный корпус

```bash
nlp-prepare --config configs/mtl.yaml
```

Источник в `configs/base.yaml` — `SlavaYus/tinysms_qa_12m`, закреплённый на конкретном commit revision. Подготовка проходит все 60 шардов только исходного `train`. Validation выделяется устойчивым хешем вопроса и seed: Unicode NFKC и сведение пробелов используются для выбора split, поэтому варианты одного вопроса попадают в один split. Сам текст вопроса сохраняется без изменений.

Byte-level BPE обучается на вопросах и коде из стратифицированной reservoir-выборки тренировочных строк всех шардов. Validation в обучение токенизатора не попадает. Существующий токенизатор можно передать через `data.tokenizer_path`.

Результат в `data/prepared` содержит токенизатор, отдельные train/validation массивы каждого шарда и `manifest.json` с количеством строк, длинами, источником, SHA-256 файлов и fingerprint корпуса. При чтении проверяются хеши, offsets, диапазон token IDs и EOS. Строки длиннее заданных лимитов отфильтровываются с учётом в manifest. Повторная подготовка в тот же каталог разрешена только для того же запроса и целых артефактов.

`data.normalize_code=false` сохраняет код дословно. При `true` допустимо статическое форматирование через AST с проверкой равенства деревьев. Код из датасета и предсказания не исполняются.

Вместо Hugging Face можно указать локальные JSONL-шарды. Каждая строка содержит строковые поля `question` и `code`; это должны быть только тренировочные данные:

```bash
nlp-prepare --config configs/ce.yaml \
  data.source.kind=local_jsonl \
  'data.source.input_paths=[data/raw/train-000.jsonl,data/raw/train-001.jsonl]' \
  data.output_dir=data/prepared_local

python train.py --config configs/ce.yaml \
  data.output_dir=data/prepared_local train.output_dir=runs/local_ce
```

### Online-подача и CPU-prefetch

`configs/online.yaml` включает `data.mode=online`, сохраняет стандартные MTL-задачи и направляет результаты в `runs/online`. Источник — исходный `train` из закреплённого Hugging Face revision либо локальные JSONL-шарды через те же `data.source.kind` и `data.source.input_paths`.

Нужен готовый токенизатор, обученный только на тренировочных данных. Его можно извлечь из совместимого checkpoint:

```bash
python scripts/export_tokenizer.py --checkpoint checkpoints/model.pt --output checkpoints/tokenizer.json
python train.py --config configs/online.yaml
```

`data.tokenizer_path` в этой конфигурации указывает на `checkpoints/tokenizer.json`. Использование его токенизатора не загружает веса: модель по умолчанию инициализируется случайно. Для старта из совместимых весов отдельно укажи `train.init_checkpoint=checkpoints/model.pt`.

Поток воспроизводимо перемешивает строки и чередует шарды, повторяя короткие. Ограниченная очередь и CPU thread pool заранее выполняют токенизацию, статические auxiliary labels и collate. При CUDA используются pinned memory и неблокирующая передача batches на GPU. Размеры буферов ограничивают опережающую подготовку, поэтому корпус целиком не хранится в памяти.

В распределённом online-режиме каждый rank читает общий исходный поток и получает свою часть глобального batch. Примеры между ranks не пересекаются, но чтение источника дублируется.

| Настройка `data.online` | По умолчанию | Назначение |
| --- | --- | --- |
| `cpu_workers` | `2` | CPU-потоки подготовки |
| `prefetch_batches` | `2` | Ёмкость очереди готовых batches |
| `shuffle_buffer` | `64` | Буфер перемешивания строк |
| `tokenization_batch_size` | `128` | Размер порции токенизации |
| `validation_scan_per_shard` | `2048` | Лимит исходных строк каждого шарда для поиска validation |

Online-validation собирается из ограниченного начала каждого исходного train-шарда по тому же хеш-правилу split. Число найденных примеров может оказаться меньше `evaluation.max_examples`; выбранные validation-строки в train не попадают. Предсказания и целевой код при подготовке не исполняются.

Online-resume восстанавливает именно потреблённую позицию: детерминированно перечитывает поток с начала до неё и отбрасывает опережающую очередь. Это сохраняет последовательность batches, но восстановление далёкой позиции может быть медленным. Проверка runtime fingerprint включает реализацию online-потока. Для смены `data.mode` начинай новое обучение, при необходимости с `train.init_checkpoint`, поскольку режим данных входит в строгую идентичность resume.

## Обучение

Конфигурации отдельного обучения наследуют `configs/base.yaml`. Дополнительные аргументы задаются в формате `key=value`. `configs/pipeline.yaml` задаёт многостадийный рецепт поверх одной из этих конфигураций.

| Конфигурация | Задачи | Градиенты общего core | Обучаемая часть |
| --- | --- | --- | --- |
| `configs/ce.yaml` | Code CE | Обычные | Весь core |
| `configs/mtl.yaml` | Code CE + structure/grounding | Строгая ортогонализация | Core и включённые головы |
| `configs/mtl_pcgrad.yaml` | Code CE + structure/grounding | PCGrad | Core и включённые головы |
| `configs/structure.yaml` | Code CE + structure | Строгая ортогонализация | Core и structure head |
| `configs/grounding.yaml` | Code CE + grounding | Строгая ортогонализация | Core и grounding head |
| `configs/decoder.yaml` | Code CE | Обычные | Decoder и его финальная нормализация |
| `configs/online.yaml` | Code CE + structure/grounding | Строгая ортогонализация | Core и включённые головы; online-подача |

```bash
python train.py --config configs/mtl.yaml
python train.py --config configs/mtl.yaml --max-updates 20
```

`--max-updates` останавливает текущий запуск после указанного числа обновлений и сохраняет checkpoint. Он не меняет `train.max_steps` и расписание learning rate: такой короткий запуск можно продолжить обычным resume.

По умолчанию каждый шаг включает Code CE и одну случайно выбранную активную дополнительную задачу. Выбор воспроизводим и управляется `task_sampling`; масштабы losses задаются отдельно в `loss_weights`.

- **Structure:** количество арифметических операций, корневая операция, глубина и число листьев живого статического DAG целевого кода. Голова использует усреднённые decoder states.
- **Grounding:** бинарные метки для decoder-позиций числовых литералов целевого кода: встречается ли соответствующее число в вопросе. Используются token offsets и статический разбор; неподдерживаемые программы и ненадёжные метки маскируются.

В режиме `orthogonal` общие градиенты задач проходят модифицированный Gram–Schmidt в воспроизводимом случайном порядке, с двумя проходами. Удаляются проекции любого знака; почти зависимые остатки обнуляются. PCGrad удаляет только компоненты, конфликтующие с исходными градиентами других задач, и не гарантирует попарной ортогональности. После обработки общие градиенты суммируются. Градиенты отдельных task heads не проецируются.

В режиме `prepared` настройка `data.sampling_mode=equal_shards` чередует шарды с одинаковой квотой и намеренно повторяет строки коротких шардов. `uniform_examples` использует общую перестановку строк без такого oversampling. `train.batch_size` задаётся на один rank; глобальный batch равен `batch_size × gradient_accumulation × world_size`. `microbatch_size` разбивает его для экономии памяти.

Два CUDA-процесса:

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/mtl.yaml
```

`train.device=auto` выбирает CUDA по local rank; без CUDA распределённое обучение использует CPU. Для явного CPU/Gloo-запуска:

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/ce.yaml \
  train.device=cpu train.precision=fp32
```

Нормировки losses и градиенты задач собираются по всем ranks до проекции. Sampler строит общий порядок строк и выдаёт ranks разные части одного глобального batch.

## Полный цикл MTL

`configs/pipeline.yaml` запускает четыре стадии с суммарной квотой 24 часа активного времени. Базовая конфигурация — `configs/online.yaml`, каталог результатов — `runs/pipeline`, исходный seed — `42`. Первая стадия по умолчанию начинает со случайных весов. Веса после обучения и их качество определяются этим запуском; рецепт не гарантирует заранее заданный результат.

| Стадия | Квота времени | Верхняя граница шагов | Warmup, шагов | Learning rate | Обучаемая часть и задачи |
| --- | ---: | ---: | ---: | ---: | --- |
| `code_warmup` | 3 ч | 100 000 | 2 000 | `3e-4` | Весь core; только Code CE, дополнительные головы выключены |
| `joint_mtl` | 18 ч | 600 000 | 4 000 | `2e-4` | Весь core и обе головы; до двух доступных задач, веса `4:1:1`, ортогонализация |
| `code_consolidation` | 2 ч | 80 000 | 1 000 | `5e-5` | Весь core; только Code CE, дополнительные головы выключены |
| `decoder` | 1 ч | 40 000 | 500 | `1e-5` | Decoder layers и `decoder_norm`; только Code CE, encoder и общие embeddings заморожены |

Стадия завершается при достижении первой из двух границ: числа шагов или её `train.max_total_seconds`. Текущий update заканчивается, сохраняется checkpoint и запускается следующая стадия. Поэтому 820 000 шагов — суммарный потолок, а не обязательное число шагов.

В `elapsed_seconds` training checkpoint накапливается активное время стадии между сессиями, включая её инициализацию и evaluation. Ожидание GPU в очереди и паузы между сессиями в квоту не входят. Это мягкий бюджет: pipeline может закончиться раньше по числу шагов или превысить 24 часа на время завершающего update, сохранения и запуска стадий. Точные сутки на конкретном Kaggle GPU не гарантируются.

После обычного warmup по шагам cosine-расписание использует большую из двух долей прогресса: по шагам и по активному времени стадии. Благодаря этому высокий потолок шагов не мешает снижению learning rate к концу временной квоты.

Сначала посмотри план без открытия датасета и модели, затем запусти обучение:

```bash
python train_pipeline.py --config configs/pipeline.yaml --plan
python train_pipeline.py --config configs/pipeline.yaml
```

В плане указаны `total_max_seconds: 86400` и `total_max_steps: 820000`, а также параметры каждой стадии.

Включённый `prepare_data: true` автоматически готовит корпус в режиме `prepared`, если `manifest.json` ещё отсутствует. В режиме `online` весь корпус предварительно не материализуется: используются исходный train-поток, готовый `data.tokenizer_path` и ограниченный validation-prefix. Токенизатор должен быть обучен только на train; его подключение не инициализирует веса модели.

Для короткой сессии с сохранением всей последовательности стадий:

```bash
python train_pipeline.py --config configs/pipeline.yaml --max-updates 20
```

Лимит относится ко всей текущей сессии, включая переходы между стадиями. Он не меняет квоты, потолки шагов и расписания learning rate. Повтор той же команды продолжает сохранённый pipeline; без `--max-updates` выполняются оставшиеся части всех стадий. `train.max_seconds` отдельно ограничивает одну сессию: такой pilot-stop сохраняет прогресс, но не завершает стадию. Её накопленное активное время продолжает учитываться при следующем запуске.

### Выбор задач и вычисления

В `joint_mtl` Code CE не обязателен на каждом шаге: `TaskSampler.sample(available)` выбирает без повторений до двух доступных задач с весами `4:1:1`. Доступность определяется наличием валидных целей в глобальном batch. Выключенные головы и задачи с полностью замаскированными целями не участвуют в выборе.

Каждая выбранная дополнительная задача независимо исключается с вероятностью `0.15`. Code CE не исключается этим механизмом. Если после исключения дополнительных задач выбор пуст, выполняется Code CE. Это исключение целой задачи на шаге; обычный dropout активаций задаётся отдельно через `model.dropout`.

На один microbatch общий backbone выполняется один раз. Вычисляются только выбранные выходные ветви: vocabulary logits нужны для Code CE, structure и grounding heads — для соответствующих задач. Градиенты каждой выбранной задачи по общему core сначала суммируются между ranks, затем ортогонализируются и складываются. Градиенты собственных параметров голов суммируются между ranks без проекции.

Все дополнительные цели выводятся статически из исходных train-вопросов и программ. Код и предсказания не исполняются. Для validation используются метрики, описанные ниже; Comet остаётся опциональным.

### Настройка стадий

В рецепте `base_config` выбирает базовую конфигурацию, `output_dir` — общий каталог, `seed` — начальный seed, а `stages` содержит имена, потолки шагов и вложенные `overrides`. Временная квота задаётся через `train.max_total_seconds` в override каждой стадии: `10800`, `64800`, `7200` и `3600` секунд соответственно. Seed стадии равен начальному seed плюс её индекс, начиная с нуля.

Например, параметры совместной стадии записываются так:

```yaml
- name: joint_mtl
  steps: 600000
  handoff: best
  overrides:
    heads:
      structure: true
      grounding: true
    task_sampling:
      always_code: false
      auxiliary_tasks_per_step: 2
      weights:
        code: 4.0
        structure: 1.0
        grounding: 1.0
      drop_probabilities:
        structure: 0.15
        grounding: 0.15
    gradients:
      mode: orthogonal
    optimizer:
      lr: 0.0002
    train:
      scope: all
      max_total_seconds: 64800
    scheduler:
      warmup_steps: 4000
```

`heads.structure` и `heads.grounding` включают параметры голов. `task_sampling.weights` управляет частотой выбора задач, а `loss_weights` — масштабом их градиентов. Для обязательного Code CE установи `always_code: true`; тогда `auxiliary_tasks_per_step` задаёт число дополнительных выбранных задач. При `always_code: false` это число выбираемых задач из всего доступного набора. `drop_probabilities` задаёт независимое исключение structure и grounding. `train.scope` принимает `all` или `decoder`; потолок шагов стадии меняется через `steps`, квота времени — через `train.max_total_seconds`, warmup — через `scheduler.warmup_steps`.

Dotlist-аргументы командной строки меняют базовую training-конфигурацию:

```bash
python train_pipeline.py --config configs/pipeline.yaml \
  train.batch_size=32 train.microbatch_size=8 comet.enabled=true
```

Порядок применения: базовая конфигурация, общие `overrides` рецепта, dotlist-аргументы, затем `overrides` конкретной стадии. Поэтому параметры heads, learning rate, выбора задач и scope, заданные в стадии, меняй в её YAML-блоке. Каталог стадии, её число шагов и seed устанавливает runner.

Для старта первой стадии из любого совместимого inference-checkpoint передай `train.init_checkpoint=checkpoints/model.pt` и подключи его токенизатор через `data.tokenizer_path`. По умолчанию `train.init_checkpoint=null`.

### Переходы, сохранение и продолжение

Рецепт использует `handoff: best`: следующая стадия получает только core предыдущего `best.pt`, если он существует, иначе `model.pt`. Для каждой новой стадии создаются новые optimizer, scheduler и включённые task heads. Внутри незавершённой стадии восстанавливается полный `last_training.pt`, включая optimizer, sampler, RNG и накопленное `elapsed_seconds`; повторная инициализация голов при таком resume не выполняется. Исчерпанная временная квота завершает стадию даже при недостигнутом потолке шагов.

Каталоги стадий находятся в `runs/pipeline/code_warmup`, `joint_mtl`, `code_consolidation` и `decoder`. В каждом сохраняются обычные checkpoints, конфигурация и логи. `runs/pipeline/pipeline.json` фиксирует идентичность рецепта, а `pipeline_summary.json` — его текущий прогресс. После завершения итоговый inference-export находится в `runs/pipeline/model.pt`.

Для продолжения сохраняй рецепт и его dotlist-аргументы. Изменения задач, модели, данных, потолков шагов или временных квот требуют нового верхнеуровневого `output_dir` рецепта. В распределённом запуске также сохраняется прежний world size:

```bash
torchrun --standalone --nproc_per_node=2 train_pipeline.py --config configs/pipeline.yaml
```

## Checkpoints и продолжение

Каталог `train.output_dir` содержит `config.json`, `run.json`, `metrics.jsonl`, `summary.json` и checkpoints:

- `last_training.pt` — модель с головами, optimizer, scheduler, AMP scaler, EMA, состояние RNG и отдельные sampler/task/projection states каждого rank.
- `model.pt` — inference export только core, его конфигурации и токенизатора; при включённом использовании EMA экспортируются EMA-веса.
- `best.pt` — inference export при улучшении validation loss, если оценка включена.

```bash
python train.py --config configs/mtl.yaml train.resume=runs/mtl/last_training.pt
```

Resume проверяет fingerprint данных, точный токенизатор, конфигурацию, world size и fingerprint реализации обучения. Из идентичности конфигурации исключены `train.resume`, `train.output_dir`, `train.init_checkpoint` и Comet-настройки. `train.init_checkpoint` служит только источником начальных весов; при resume веса берутся из полного training checkpoint. В частности, `train.max_steps`, AMP, модель, задачи и optimizer должны совпадать. Восстанавливается позиция действительно потреблённых batches, а не опережающей загрузки workers. Training checkpoint загружается как доверенный локальный файл; inference export для resume не подходит.

Для нового обучения из inference-весов используй `train.init_checkpoint`. Это загрузка core с новым optimizer и новым расписанием. Архитектура и токенизатор должны быть совместимы. Для подготовки нового корпуса с токенизатором checkpoint:

```bash
python scripts/export_tokenizer.py --checkpoint checkpoints/model.pt --output tokenizer.json
nlp-prepare --config configs/decoder.yaml \
  data.tokenizer_path=tokenizer.json data.output_dir=data/prepared_compatible
python train.py --config configs/decoder.yaml \
  data.output_dir=data/prepared_compatible train.init_checkpoint=checkpoints/model.pt
```

В `decoder.yaml` encoder и общие embeddings заморожены; обучаются decoder layers и `decoder_norm`. `train.resume` и `train.init_checkpoint` взаимоисключающие.

Validation считает `validation/loss`, `validation/token_accuracy` и `validation/sequence_exact_match_at_1` — точное текстовое совпадение с декодированной ссылкой. Исполнение программ и семантический pass@1 не измеряются.

## Comet

Comet отключён по умолчанию. Для online-логирования задай `COMET_API_KEY` в окружении или локальном `.env` и включи `comet.enabled=true`. Значение ключа в YAML не требуется. Offline-режим не требует ключа:

```bash
python train.py --config configs/mtl.yaml comet.enabled=true comet.offline=true
```

Настройки проекта, workspace и offline-каталога находятся в секции `comet`. При resume используется сохранённый experiment key; отсутствие поддержки продолжения в SDK вызывает ошибку. Логирует только rank 0. Автоматический сбор окружения, Git, кода и stdout отключён; секретные поля параметров скрываются. Загрузка весов как assets включается отдельно через `comet.log_checkpoints=true`.

## Inference

```bash
nlp-predict --checkpoint runs/mtl/model.pt \
  --test data/test.jsonl --output outputs/predictions.jsonl
```

Стандартный режим — FP32, beam size 5, batch size 32 и `cuda:0`. Для CPU добавь `--device cpu`. Вход читается только из указанного локального JSONL; нужны строковые `question`. Выход содержит исходные `question` и сгенерированные `code` в исходном порядке. Вопросы передаются модели дословно. Рядом сохраняется `.metrics.json` с параметрами запуска, временем, количеством строк и SHA-256 результата после `fsync` и readback.

`inference.ipynb` загружает код инференса из [GitHub](https://github.com/denakifev/nlp-codegen-transformer) на закреплённом commit и `model.pt` из [публичной папки Google Drive](https://drive.google.com/drive/folders/1MMRr-wJ8YEaTPw1_YZeXrJm6v1KbjQTH). Проверяются SHA-256 исходников и checkpoint. Токенизатор и конфигурация архитектуры уже находятся внутри `model.pt`; отдельно скачивать `tokenizer.json` для inference не нужно.

Для запуска в Kaggle:

1. Включи **GPU T4** и **Internet**.
2. Подключи dataset с `test.jsonl`, содержащим строковое поле `question` в каждой строке.
3. Нажми **Run All**. При нескольких найденных `test.jsonl` укажи нужный `TEST_PATH` в первой ячейке.
4. Скачай `/kaggle/working/predictions.jsonl` из Outputs.

Ноутбук устанавливает только зависимости инференса и скачивания, использует предустановленный PyTorch и кэширует проверенные файлы. Основной замер — `generation_seconds`: генерация на загруженной модели после токенизации, с синхронизацией CUDA перед началом и после завершения. Скачивание, установка, загрузка checkpoint и токенизация в него не входят. Время подготовки и `total_seconds` сохраняются отдельно; последнее также включает загрузку, импорты и запись результата. Для замера в новом процессе перезапусти kernel.

Если обученные веса недоступны или не прошли проверку, inference пропускается. Последняя ячейка проверяет ту же архитектуру со случайными весами на синтетическом батче: 32 последовательности по 128 входных токенов, до 64 новых токенов, FP32 beam5. Это технический замер, а не оценка качества или времени полного test-инференса. Он сохраняет только `random_benchmark.metrics.json` и не записывает случайные программы в `predictions.jsonl`.

## Проверки

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Тесты используют небольшие локальные корпуса и модели. Проверка Gloo запускает два CPU-процесса и требует разрешённого соединения через loopback. Comet проверяется поддельным SDK без подключения к сервису.
