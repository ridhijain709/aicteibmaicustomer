# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application file into the working directory
COPY app.py .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py"]

print("Dockerfile created successfully.")
