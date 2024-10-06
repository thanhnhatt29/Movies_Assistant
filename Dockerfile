FROM python:3.11-slim

WORKDIR /app

RUN pip install pipenv

COPY data/movies.csv data/movies.csv
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --ignore-pipfile --system

COPY movies_assistant .

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:5000 app:app