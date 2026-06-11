import os, sys, cv2, time, io, threading, re, mimetypes
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session, Response
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from detection.lane_cv import LaneCVDetector
from web.db import init_db, add_detection, get_all_detections, get_detection, delete_detection
from web.db import search_detections, update_detection, clear_all
from web.db import export_csv, export_log_csv, get_stats, log_action

# 预先加载深度学习模块，避免请求时加载
try:
    from detection.lane_dl import detect_dl, _get_model
    print("Deep learning module loaded successfully")
    # 预先初始化模型
    model, device = _get_model()
    print(f"Model loaded on device: {device}")
except Exception as e:
    import traceback
    print(f"Warning: Failed to preload DL module: {e}")
    print(traceback.format_exc())
    detect_dl = None

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(WEB_DIR, 'templates'),
            static_folder=os.path.join(PROJECT_ROOT, 'static'))
app.secret_key = 'lane_detection_secret_2026'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'uploads')
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def url_path(absolute_path):
    rel = os.path.relpath(absolute_path, PROJECT_ROOT)
    return rel.replace('\\', '/')

@app.route('/')
def index():
    stats = get_stats()
    detections = get_all_detections(limit=20)
    return render_template('index.html', stats=stats, detections=detections)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    method = request.form.get('method', 'cv')
    draw_windows = request.form.get('draw_windows') == '1'

    ext = os.path.splitext(file.filename)[1].lower()
    ts = int(time.time() * 1000)

    if ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        return process_image(file, method, draw_windows, ts)
    elif ext in ('.mp4', '.avi', '.mov', '.mkv'):
        return process_video(file, method, draw_windows, ts)
    else:
        return 'Unsupported format', 400

def process_image(file, method, draw_windows, ts):
    try:
        filename = f'{ts}_{file.filename}'
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        img = cv2.imread(upload_path)
        if img is None:
            return 'Failed to read image', 400

        result, detected, curves, left_fit, right_fit, dl_lanes = run_detection(img, method, draw_windows)

        result_path = os.path.join(RESULT_FOLDER, f'r_{filename}')
        cv2.imwrite(result_path, result)

        if dl_lanes is not None:
            lanes_count = dl_lanes
            left_curve = left_fit
            right_curve = right_fit
        else:
            left_fitx, right_fitx, ploty = curves
            left_curve = left_fit[0] if left_fit is not None else None
            right_curve = right_fit[0] if right_fit is not None else None
            lanes_count = 0
            if left_fitx is not None:
                lanes_count += 1
            if right_fitx is not None:
                lanes_count += 1

        det_id = add_detection(file.filename, method.upper(), lanes_count, detected,
                               left_curve, right_curve, url_path(upload_path), url_path(result_path))
        return redirect(url_for('result', det_id=det_id))
    except Exception as e:
        import traceback
        error_msg = f"process_image error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f'Error processing image: {str(e)}', 500

def process_video(file, method, draw_windows, ts):
    filename = f'{ts}_{file.filename}'
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    cap = cv2.VideoCapture(upload_path)
    detector = None
    if method == 'cv':
        detector = LaneCVDetector(smooth_window=5)
    else:
        from detection.lane_dl import LaneDLDetector
        detector = LaneDLDetector(smooth_window=5)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    h, w = 590, 1640

    out_filename = f'r_{ts}_{os.path.splitext(file.filename)[0]}.mp4'
    result_path = os.path.join(RESULT_FOLDER, out_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(result_path, fourcc, min(fps, 20), (w, h))

    total_detected = 0
    max_frames = min(frame_count, 200)
    preview_saved = False
    preview_path = ''
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (w, h))
        result, detected, _, _, _, _ = run_detection(frame, method, draw_windows, detector)
        if detected:
            total_detected += 1
        if not preview_saved:
            preview_filename = f'preview_{ts}_{os.path.splitext(file.filename)[0]}.jpg'
            preview_path = os.path.join(RESULT_FOLDER, preview_filename)
            cv2.imwrite(preview_path, result)
            preview_saved = True
        vout.write(result)

    cap.release()
    vout.release()

    det_id = add_detection(file.filename, method.upper(), 2, total_detected > 0,
                           None, None, url_path(upload_path), url_path(result_path),
                           url_path(preview_path) if preview_path else None)
    return redirect(url_for('result', det_id=det_id))

def run_detection(img, method, draw_windows, detector=None):
    if method == 'cv':
        if detector is None:
            detector = LaneCVDetector(smooth_window=5)
        result, detected, curves = detector.detect(img, draw_windows)
        left_fit = detector.left_fit
        right_fit = detector.right_fit
        return result, detected, curves, left_fit, right_fit, None
    else:
        if detect_dl is None:
            return img.copy(), False, (None, None, None), None, None, 0
        try:
            result, detected, curves, left_fit, right_fit, dl_lanes = detect_dl(img)
            return result, detected, curves, left_fit, right_fit, dl_lanes
        except Exception as e:
            import traceback
            error_msg = f"DL detection error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return img.copy(), False, (None, None, None), None, None, 0

@app.route('/result/<int:det_id>')
def result(det_id):
    current = get_detection(det_id)
    detections = get_all_detections(100)
    return render_template('result.html', detections=detections, current=current)

# ========== RESTful CRUD API ==========

@app.route('/api/detections')
def api_list():
    keyword = request.args.get('keyword', '')
    method = request.args.get('method', '')
    detected = request.args.get('detected', '')
    page = int(request.args.get('page', 1))
    per = 20
    rows, total = search_detections(keyword, method, detected, (page - 1) * per, per)
    return jsonify({'rows': rows, 'total': total})

@app.route('/api/detection/<int:det_id>', methods=['GET'])
def api_get(det_id):
    d = get_detection(det_id)
    return jsonify(d) if d else ('Not found', 404)

@app.route('/api/detection/<int:det_id>', methods=['PUT'])
def api_update(det_id):
    data = request.get_json()
    update_detection(
        det_id,
        filename=data.get('filename'),
        lanes_count=data.get('lanes_count'),
        detected=data.get('detected'),
        curve_left=data.get('curve_left'),
        curve_right=data.get('curve_right'),
    )
    return jsonify({'ok': True})

@app.route('/api/detection/<int:det_id>', methods=['DELETE'])
def api_delete(det_id):
    delete_detection(det_id)
    return jsonify({'ok': True})

@app.route('/delete/<int:det_id>', methods=['POST'])
def delete(det_id):
    delete_detection(det_id)
    return redirect(url_for('index'))

@app.route('/clear', methods=['POST'])
def clear():
    clear_all()
    return redirect(url_for('index'))

@app.route('/export')
def export():
    path, count = export_csv()
    log_action('EXPORT_CSV', f'Exported {count} records')
    return send_file(path, as_attachment=True, download_name=os.path.basename(path), mimetype='text/csv')

@app.route('/export_log')
def export_log():
    path, count = export_log_csv()
    return send_file(path, as_attachment=True, download_name=os.path.basename(path), mimetype='text/csv')

@app.route('/stats')
def stats_api():
    return jsonify(get_stats())

@app.route('/video/<path:filepath>')
def serve_video(filepath):
    full_path = os.path.join(PROJECT_ROOT, filepath)
    if not os.path.exists(full_path):
        return 'Not found', 404

    file_size = os.path.getsize(full_path)
    range_header = request.headers.get('Range', None)

    if range_header:
        byte1, byte2 = 0, None
        m = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if m:
            byte1 = int(m.group(1))
            if m.group(2):
                byte2 = int(m.group(2))

        if byte2 is None:
            byte2 = file_size - 1

        length = byte2 - byte1 + 1
        with open(full_path, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)

        rv = Response(data, 206, mimetype='video/mp4',
                      direct_passthrough=True)
        rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
        rv.headers.add('Accept-Ranges', 'bytes')
        rv.headers.add('Content-Length', str(length))
        return rv

    return send_file(full_path, mimetype='video/mp4', conditional=True)

if __name__ == '__main__':
    init_db()
    log_action('START', 'Application started')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
