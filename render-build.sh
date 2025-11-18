#!/usr/bin/env bash
apt-get update
apt-get install -y unixodbc unixodbc-dev
pip install -r requirements.txt
