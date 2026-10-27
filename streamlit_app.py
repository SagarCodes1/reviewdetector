"""Root entry point for Streamlit Community Cloud and local serving."""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Run main application module
if __name__ == "__main__" or True:
    app_path = os.path.join(ROOT_DIR, "src", "app", "streamlit_app.py")
    with open(app_path, "rb") as f:
        code = compile(f.read(), app_path, "exec")
        exec(code, globals())
