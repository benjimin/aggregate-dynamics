#!/bin/env bash
export FLASK_APP=server.py
export FLASK_DEBUG=1
bash -c "sleep 5; xdg-open http://127.0.0.1:5000/" &
flask run 
