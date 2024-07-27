# Use Python base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy to the working directory
COPY . /app/
RUN apt-get update && \
    apt-get install -y wget
# Install any required packages
RUN pip install --no-cache-dir -r requirements.txt
# Install Kaggle API
RUN pip install kaggle

# Copy the kaggle.json file to the container
COPY kaggle.json /root/.kaggle/kaggle.json

# Ensure the kaggle.json file has the correct permissions
RUN chmod 600 /root/.kaggle/kaggle.json

# Command to run model.py
CMD ["bash", "-c", "if [ '$CLEAR_DATASETS' = 'true' ]; then rm -rf /data/*; fi && python3 model.py"]
