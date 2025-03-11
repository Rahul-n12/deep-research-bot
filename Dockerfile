# Use the base Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Copy the .env file
COPY .env /app/.env

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8080

# Set environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8080

# Command to run the Streamlit app
CMD ["streamlit", "run", "bot.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
