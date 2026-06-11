import sqlite3
import datetime
import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'lane_detection.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        method TEXT NOT NULL,
        lanes_count INTEGER,
        detected INTEGER,
        curve_left REAL,
        curve_right REAL,
        upload_path TEXT,
        result_path TEXT,
        created_at TEXT
    )''')
    # migration: add upload_path column if upgrading from old schema
    try:
        c.execute('ALTER TABLE detections ADD COLUMN upload_path TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        c.execute('ALTER TABLE detections ADD COLUMN preview_path TEXT')
    except sqlite3.OperationalError:
        pass
    c.execute('''CREATE TABLE IF NOT EXISTS operation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        detail TEXT,
        created_at TEXT
    )''')
    conn.commit()
    conn.close()

def log_action(action, detail=''):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO operation_log (action, detail, created_at) VALUES (?, ?, ?)',
              (action, detail, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def add_detection(filename, method, lanes_count, detected, curve_left, curve_right, upload_path, result_path, preview_path=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO detections (filename, method, lanes_count, detected, curve_left, curve_right, upload_path, result_path, preview_path, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (filename, method, lanes_count, 1 if detected else 0,
               round(curve_left, 8) if curve_left else None,
               round(curve_right, 8) if curve_right else None,
               upload_path, result_path, preview_path,
               datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    log_action('DETECT', f'{filename} ({method}), lanes={lanes_count}, detected={detected}')
    return row_id

def get_all_detections(limit=100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY id DESC LIMIT ?', (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_detection(det_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM detections WHERE id=?', (det_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def search_detections(keyword='', method='', detected='', offset=0, limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    sql = 'SELECT * FROM detections WHERE 1=1'
    params = []
    if keyword:
        sql += ' AND filename LIKE ?'
        params.append(f'%{keyword}%')
    if method:
        sql += ' AND method=?'
        params.append(method)
    if detected != '' and detected is not None:
        sql += ' AND detected=?'
        params.append(int(detected))
    c.execute(f'SELECT COUNT(*) FROM ({sql})', params)
    total = c.fetchone()[0]
    sql += ' ORDER BY id DESC LIMIT ? OFFSET ?'
    params.extend([limit, offset])
    c.execute(sql, params)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows, total

def update_detection(det_id, filename=None, lanes_count=None, detected=None, curve_left=None, curve_right=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    updates = []
    params = []
    if filename is not None:
        updates.append('filename=?')
        params.append(filename)
    if lanes_count is not None:
        updates.append('lanes_count=?')
        params.append(lanes_count)
    if detected is not None:
        updates.append('detected=?')
        params.append(1 if detected else 0)
    if curve_left is not None:
        updates.append('curve_left=?')
        params.append(round(curve_left, 8))
    if curve_right is not None:
        updates.append('curve_right=?')
        params.append(round(curve_right, 8))
    if updates:
        params.append(det_id)
        c.execute(f'UPDATE detections SET {", ".join(updates)} WHERE id=?', params)
        conn.commit()
    conn.close()
    log_action('UPDATE', f'id={det_id}, fields={updates}')
    return True

def clear_all():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT upload_path, result_path FROM detections')
    files = c.fetchall()
    c.execute('DELETE FROM detections')
    conn.commit()
    conn.close()
    for row in files:
        for path in row:
            abs_path = os.path.join(PROJECT_ROOT, path) if path and not os.path.isabs(path) else path
            if path and os.path.exists(abs_path):
                try:
                    os.remove(abs_path)
                except OSError:
                    pass
    log_action('CLEAR', f'Deleted {len(files)} records')
    return len(files)

def delete_detection(det_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT filename, upload_path, result_path FROM detections WHERE id=?', (det_id,))
    row = c.fetchone()
    c.execute('DELETE FROM detections WHERE id=?', (det_id,))
    conn.commit()
    conn.close()
    if row:
        log_action('DELETE', f'id={det_id}, file={row[0]}')
        for path in (row[1], row[2]):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    return True

def export_csv(output_path=None):
    if output_path is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(os.path.dirname(__file__), f'export_detections_{ts}.csv')
    else:
        output_path = os.path.join(os.path.dirname(__file__), output_path)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY id DESC')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    if rows:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            f.write('id,filename,method,lanes_count,detected,curve_left,curve_right,upload_path,result_path,created_at\n')
    log_action('EXPORT_CSV', f'{len(rows)} records -> {output_path}')
    return output_path, len(rows)

def export_log_csv(output_path=None):
    if output_path is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(os.path.dirname(__file__), f'export_operation_log_{ts}.csv')
    else:
        output_path = os.path.join(os.path.dirname(__file__), output_path)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM operation_log ORDER BY id DESC')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    if rows:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            f.write('id,action,detail,created_at\n')
    return output_path, len(rows)

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*), SUM(detected) FROM detections')
    total, detected = c.fetchone()
    c.execute('SELECT method, COUNT(*) FROM detections GROUP BY method')
    methods = c.fetchall()
    conn.close()
    return {
        'total': total or 0,
        'detected': detected or 0,
        'rate': round(detected / total * 100, 1) if total else 0,
        'methods': dict(methods) if methods else {}
    }
