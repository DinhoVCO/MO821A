# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY client_requirements.txt /app/client_requirements.txt
COPY client.py /app/client.py
COPY model.py /app/model.py
COPY load_data.py  /app/load_data.py


# Install gcc and other dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install -r client_requirements.txt


