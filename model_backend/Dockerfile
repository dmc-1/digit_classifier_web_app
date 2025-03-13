FROM pytorch/pytorch

WORKDIR /app

COPY . /app

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "model_backend.server.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]