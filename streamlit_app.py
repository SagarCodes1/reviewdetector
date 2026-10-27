"""Root entry point for Streamlit Community Cloud and local serving."""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import src.app.streamlit_app as app

app.main()
