#!/usr/bin/env bash
python3 --version
apt-get update
apt-get install -y unixodbc unixodbc-dev g++ build-essential
pip install --upgrade pip
pip install -r requirements.txt
