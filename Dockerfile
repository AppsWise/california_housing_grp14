
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -ms /bin/bash appuser

# Create the log file and set permissions
RUN touch /var/log/app.log && chown appuser:appuser /var/log/app.log

# Switch to the non-root user
USER appuser

# Make port 80 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "src/app.py"]

