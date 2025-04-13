FROM python:3.13-slim

WORKDIR /app

RUN useradd -ms /bin/bash nonroot

USER nonroot

ENV PATH="/home/nonroot/.local/bin:$PATH"

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

EXPOSE 9000

CMD ["fastapi", "run", "main.py", "--port", "9000"]
