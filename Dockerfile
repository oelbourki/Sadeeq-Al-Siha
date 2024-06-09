# Use the official Python image from the Docker Hub
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./
# COPY public ./
# COPY data ./
# COPY ingest.py ./
# COPY  app-qa.py ./
# COPY  chainlit.md ./

RUN pip install --upgrade pip

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .
# RUN python Sadeeq-Al-Siha/ingest.py
# Expose the port the app runs on
EXPOSE 8005

# Define the command to run the application
CMD ["chainlit", "run", "app-qa.py", "--host", "0.0.0.0", "--port", "8005"]
