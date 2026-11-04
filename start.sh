#!/bin/sh
set -e
python3 fig47/static_server.py &
exec python3 server_128.py
