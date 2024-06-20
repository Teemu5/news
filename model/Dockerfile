# Use Python base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy to the working directory
COPY . /app/

# Install any required packages
RUN pip install --no-cache-dir -r requirements.txt

# Command to run model.py
CMD ["python3", "model.py"]
