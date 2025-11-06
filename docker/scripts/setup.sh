#!/bin/bash

# ============================================
# QKD System Docker Setup Script
# ============================================

echo "QKD System Docker Setup"
echo "======================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created. Please edit it with your settings."
else
    echo "✓ .env file already exists"
fi

# Pull Ollama model inside container
echo ""
echo "Pulling Ollama model (this will happen in container)..."
echo "Note: The model download will occur when the Ollama container starts"

# Start services
echo ""
echo "Starting Docker services..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for MySQL to be ready
echo ""
echo "Waiting for MySQL to be ready..."
sleep 30

# Check service status
echo ""
echo "Checking service status..."
docker-compose -f docker-compose.dev.yml ps

echo ""
echo "Setup complete!"
echo ""
echo "Services should be available at:"
echo "  - Flask GUI: http://localhost:5000"
echo "  - MySQL: localhost:3307"
echo "  - Ollama API: http://localhost:11434"
echo ""
echo "To view logs: docker-compose -f docker-compose.dev.yml logs -f"
echo "To stop services: docker-compose -f docker-compose.dev.yml down"