# Use Python 3.10 devcontainer base image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.10-bullseye

# Install uv from the official Docker image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set up environment
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONPATH="/workspaces/practical-python-for-time-series-analysis"

# Create working directory
WORKDIR /workspace

# Copy project file
COPY pyproject.toml ./

# Install dependencies with caching for better performance
# uv sync will create the virtual environment automatically
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --compile-bytecode --no-dev

# Set ownership of virtual environment to vscode user
RUN chown -R vscode:vscode /opt/venv

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:${PATH}"

# Switch to vscode user and update bashrc
USER vscode
RUN echo 'export PATH="/opt/venv/bin:${PATH}"' >> ~/.bashrc

USER root