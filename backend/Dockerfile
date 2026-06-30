# Use a slim version of Python for a smaller image size
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install System Dependencies
# Your code uses 'psycopg' and 'PyMuPDF'. These require C-compilers and 
# PostgreSQL dev headers to install correctly from requirements.txt.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
# We copy this first to take advantage of Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Your Full Code
# This copies your 1664-line main.py and the entire /src folder.
COPY . .

# 4. Fix the "ModuleNotFoundError"
# This is the most important line. It tells Python that the root (/app)
# and your logic folder (/app/src) are both valid import paths.
ENV PYTHONPATH=/app:/app/src

# 5. Runtime Configuration
EXPOSE 8000

# We use uvicorn to serve your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]