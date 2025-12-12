FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts

RUN pip install --no-cache-dir .

EXPOSE 5000

CMD ["uvicorn", "base_agent.api:app", "--host", "0.0.0.0", "--port", "5000"]
