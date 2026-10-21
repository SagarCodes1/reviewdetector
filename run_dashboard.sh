#!/bin/bash
# Shell script to launch the Streamlit Fake Review Detection Dashboard

export PYTHONPATH="/Users/sagar0908/Projects:$PYTHONPATH"
streamlit run /Users/sagar0908/Projects/src/app/streamlit_app.py --server.port 8501 --server.headless true
