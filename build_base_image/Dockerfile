FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app/
# Install necessary tools
RUN apt-get update && \
    apt-get install -y wget unzip curl git python3-pip
# Install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Convert the Jupyter notebook to a Python script and run it
RUN pip3 install nbconvert
