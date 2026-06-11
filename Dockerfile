# Loan Compliance Service — container recipe.
# Build:  docker build -t loan-compliance:local .
# Run:    docker run --rm -p 8000:8000 loan-compliance:local
#
# Layer order is deliberate (cache-friendly): things that change rarely
# come first; things that change often (your code) come last.

FROM python:3.12-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

RUN python -m spacy download en_core_web_sm

COPY rules/ ./rules/
COPY backend/ ./backend/

# Bake demo data into the image: seeded portfolio + policy corpus.
# Containers start demo-ready; policy store uses file mode until
# DATABASE_URL points at postgres.
RUN cd backend \
    && python ml_training/synthetic_data.py --n 500 \
    && python rag/generate_corpus.py

WORKDIR /app/backend

EXPOSE 8000

# 0.0.0.0, not 127.0.0.1: inside a container, localhost means "this
# container only" — 0.0.0.0 lets mapped/ingress traffic reach it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]