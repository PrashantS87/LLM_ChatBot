# Use lightweight Python image
FROM python:3.12.8

# Set working directory
WORKDIR /app

# Copy code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (for clarity)
EXPOSE 8080

# Start app using Gunicorn (recommended for production)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
