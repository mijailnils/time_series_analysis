#!/bin/bash

# The virtual environment is already created at /opt/venv by the Dockerfile
# We just need to ensure it's in the PATH and install the Jupyter kernel

# Add the virtual environment to PATH
export PATH="/opt/venv/bin:$PATH"

# Also add to bashrc for persistence
echo 'export PATH="/opt/venv/bin:$PATH"' >> ~/.bashrc

# Install Jupyter kernel using the virtual environment's Python
echo "Installing IPython kernel..."
/opt/venv/bin/python -m ipykernel install --user --name time-series --display-name "Python (Time Series)"

# Set up IPython startup configuration
echo "Setting up IPython configuration..."
mkdir -p ~/.ipython/profile_default/startup && cp .devcontainer/pandas-startup.py ~/.ipython/profile_default/startup/00-pandas-options.py

echo "Devcontainer setup complete!"