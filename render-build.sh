#!/usr/bin/env bash
# Render build script with Fortran compiler

echo "🔧 Installing build dependencies..."
apt-get update && apt-get install -y gfortran

echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
