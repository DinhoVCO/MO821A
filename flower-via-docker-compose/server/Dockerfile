# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt
COPY server.py /app/server.py
COPY strategy.py /app/strategy.py


# Install gcc and other dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt


