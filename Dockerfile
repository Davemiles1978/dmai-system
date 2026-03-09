FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Make scripts executable
RUN chmod +x *.sh *.py

# Create entrypoint script that finds free ports
RUN echo '#!/bin/bash\n\
python /app/port_scanner.py\n\
WEB_PORT=$(python -c "from port_scanner import PortScanner; print(PortScanner().find_free_port(8080))")\n\
API_PORT=$(python -c "from port_scanner import PortScanner; print(PortScanner().find_free_port(8889))")\n\
echo "🚀 Starting on ports: Web=$WEB_PORT, API=$API_PORT"\n\
python /app/api_server.py --port $API_PORT &\n\
python -m http.server $WEB_PORT --bind 0.0.0.0 --directory /app/ui\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose range of ports (documentation only)
EXPOSE 8000-9000

# Start with dynamic port detection
CMD ["/app/start.sh"]
