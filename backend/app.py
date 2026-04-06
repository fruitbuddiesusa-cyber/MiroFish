"""
Synthetic Data Engine — Application Entry Point
"""

import os
import sys
import warnings

# Windows UTF-8 fix
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Suppress third-party warnings
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config.settings import Settings


def main():
    """Main entry point"""
    errors = Settings.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease check your .env file")
        sys.exit(1)

    app = create_app()

    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Settings.DEBUG

    print(f"\n{'='*50}")
    print(f"  Synthetic Data Engine")
    print(f"  Running on http://{host}:{port}")
    print(f"  Debug: {debug}")
    print(f"{'='*50}\n")

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()
