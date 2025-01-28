# Base Image
FROM python:alpine3.21
# Set working directory
WORKDIR /app

# Copy necessary files
COPY Main.py /app/app.py
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN apk add --no-cache \
    build-base python3 && \
    pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 9000

# Command to run the FastAPI server
CMD ["fastapi", "run", "app.py", "--port", "9000"]
