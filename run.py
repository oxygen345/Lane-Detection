import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.app import app
from web.db import init_db, log_action

if __name__ == '__main__':
    init_db()
    log_action('START', 'Application started')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
