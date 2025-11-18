#!/usr/bin/env bash
apt-get update
apt-get install -y curl apt-transport-https gnupg
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev

# Fix missing libodbc.so.2 error
ln -s /usr/lib/x86_64-linux-gnu/libodbc.so.2.0.0 /usr/lib/x86_64-linux-gnu/libodbc.so.2 2>/dev/null || true
