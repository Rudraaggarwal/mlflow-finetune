###################################################################
# 1. Base image and essential tools
###################################################################
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
        curl \
        cron \
    && rm -rf /var/lib/apt/lists/*

###################################################################
# 2. Guarantee “python”/“pip” commands and install uv
###################################################################
# Slim images have python3/pip3 but no “python” or “pip” symlinks.
RUN ln -sf $(which python3) /usr/local/bin/python
RUN ln -sf $(which pip3)    /usr/local/bin/pip

# Install uv and make sure its user-local install dir is on $PATH
RUN pip install uv
RUN pip install requests
RUN pip install python-dotenv

ENV PATH="/root/.local/bin:/usr/local/bin:${PATH}"

###################################################################
# 3. Resolve Python dependencies (README needed by Hatchling)
###################################################################
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen

###################################################################
# 4. Copy application source
###################################################################
COPY . .

###################################################################
# 6. Runtime settings
###################################################################
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uv", "run", "app/__main__.py"]