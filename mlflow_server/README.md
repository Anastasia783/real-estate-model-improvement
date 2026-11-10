
# MLflow Tracking Server и Model Registry

## Среда выполнения


В связи с недоступностью учебной виртуальной машины проект выполняется локально на Windows.

Команды в инструкции приведены для PowerShell. Python-код запускается в виртуальном окружении, а MLflow и PostgreSQL — в Docker-контейнерах. Для хранения артефактов MLflow используется облачное S3-хранилище Yandex Cloud. 

Shell-скрипт запуска MLflow сохранён в репозитории в соответствии с требованиями задания.
Для локального запуска из PowerShell используются команды Docker Compose.




## Состав директории

- start_mlflow.sh — shell-скрипт запуска сервисов
-compose.yaml — конфигурация PostgreSQL и MLflow
- Dockerfile — сборка образа MLflow
- requirements.txt — зависимости серверного контейнера
- .env.example — шаблон переменных окружения
- register_baseline.py — логирование и регистрация baseline
- init/01-create-databases.sql — SQL-файл инициализации PostgreSQL

## Подготовка окружения

Команды PowerShell выполняются из корня репозитория.
Создайте файл настроек на основе шаблона, если `.env` ещё отсутствует: 
 для owershell
Copy-Item .\mlflow_server\.env.example .\mlflow_server\.env


Заполните переменные окружения:

- `MLFLOW_TRACKING_URI` — адрес Tracking Server для Python-клиента
- `MLFLOW_S3_ENDPOINT_URL` — endpoint S3
- `AWS_ACCESS_KEY_ID` — идентификатор ключа доступа
- `AWS_SECRET_ACCESS_KEY` — секретный ключ
- `MLFLOW_ARTIFACT_ROOT` — корень артефактов

Для локального Python-клиента: MLFLOW_TRACKING_URI=http://localhost:5000
```

## Запуск из PowerShell

Из корня репозитория:

 для powershell
docker compose --project-directory .\mlflow_server --env-file .\mlflow_server\.env -f .\mlflow_server\compose.yaml up --build --wait --wait-timeout 180


Проверка состояния:

 для powershell
docker compose --project-directory .\mlflow_server -f .\mlflow_server\compose.yaml ps


Интерфейс MLflow: http://localhost:5000


## Запуск через shell-скрипт

Из корня репозитория в Bash, при доступном Docker Compose:  bash mlflow_server/start_mlflow.sh


Скрипт переходит в собственную директорию и запускает сервисы через Docker Compose.

## Хранение метаданных и артефактов

Артефакты хранятся в S3. Сервер запускается с параметрами:

--backend-store-uri — подключение к PostgreSQL;
--default-artifact-root — корень артефактов;
--no-serve-artifacts — прямой доступ клиентов к S3.

Поэтому переменные доступа к S3 должны быть доступны не только серверу, но и Python-клиенту, который логирует или загружает артефакты.

## Регистрация исходной baseline-модели 

Скрипт `register_baseline.py` запускается из Python-окружения проекта, содержащего зависимости для загрузки исходной модели.

Необходимы исходные файлы:

baseline/models/model.joblib
baseline/data/test.parquet
baseline/metrics/metrics.json
директория baseline/metadata

Из корня репозитория: для powershell .\mlflow_server\register_baseline.py


Скрипт загружает готовую модель без переобучения,  пересчитывает тестовые метрики и сравнивает их с исходным отчётом.

Затем логирует параметры, метрики, метаданные, код, модель, сигнатуру, пример входных данных и версии зависимостей.

## Сохранённые результаты

Эксперимент:

- имя: `real_estate_model_improvement`
- Experiment ID: `2`

Baseline для дальнейшего сравнения:

- имя модели: `real_estate_price`;
- версия: `2`
- Run ID: `c9072133902b404a9e52fb4ee64f276a`
- Registry URI: `models:/real_estate_price/2`

Идентификаторы относятся к текущему развёртыванию.
При создании проекта в новом хранилище они будут другими.

## Просмотр логов 
docker compose --project-directory .\mlflow_server -f .\mlflow_server\compose.yaml logs --tail 100 mlflow


## Остановка сервисов
docker compose --project-directory .\mlflow_server -f .\mlflow_server\compose.yaml stop

