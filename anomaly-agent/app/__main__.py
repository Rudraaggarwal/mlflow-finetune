"""
Main entry point for the Anomaly Agent.
Now uses the unified server that combines A2A and FastAPI webhook capabilities.
"""

from app.unified_server import main

if __name__ == '__main__':
    main()