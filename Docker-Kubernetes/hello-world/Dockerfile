FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY . /app

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]