FROM python:3.11-slim AS builder

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv venv /venv && \
    . /venv/bin/activate && \
    uv pip install --upgrade pip && \
    uv pip install .

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /venv /venv

COPY . .

ENV PATH="/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]