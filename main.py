import os
import sys
import logging
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel

UPLOAD_FOLDER = 'uploads'
TRANSCRIBE_FOLDER = 'transcribes'
LOG_FOLDER = 'log'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 進捗状態とキャンセルフラグの管理
PROGRESS = {}
CANCEL_FLAGS = {}

# ベースURL設定 (デフォルトは空文字列)
BASE_URL = ''

# システムプレフィックスのURL処理
class ReverseProxied(object):
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]

        # /system/whisper の処理
        if environ['PATH_INFO'].startswith('/system/whisper'):
            path_info = environ['PATH_INFO']
            environ['PATH_INFO'] = path_info[len('/system/whisper'):] or '/'
            global BASE_URL
            BASE_URL = '/system/whisper'
            logging.info(f"URLパス変換: {path_info} -> {environ['PATH_INFO']} (BASE_URL: {BASE_URL})")

        return self.app(environ, start_response)

app.wsgi_app = ReverseProxied(app.wsgi_app)

for folder in [UPLOAD_FOLDER, TRANSCRIBE_FOLDER, LOG_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_log_path():
    now = datetime.now().strftime('%Y%m%d%H%M')
    return os.path.join(LOG_FOLDER, f'{now}.log')

# ログ設定
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler(get_log_path(), encoding='utf-8')
file_handler.setFormatter(log_formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logging.getLogger().handlers = []
logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(stream_handler)
logging.getLogger().setLevel(logging.INFO)

PROGRESS = {}
CANCEL_FLAGS = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_audio_duration(audio_path):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        import wave
        import contextlib
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    elif ext in [".m4a", ".mp3"]:
        try:
            from pydub.utils import mediainfo
            info = mediainfo(audio_path)
            return float(info["duration"])
        except Exception as e:
            logging.error(f"音声長取得エラー: {e}")
            return 0
    else:
        try:
            import soundfile as sf
            f = sf.SoundFile(audio_path)
            return len(f) / f.samplerate
        except Exception as e:
            logging.error(f"音声長取得エラー: {e}")
            return 0

MODEL_SIZE = "large-v2"
whisper_model = WhisperModel(MODEL_SIZE, device="auto", compute_type="auto")

@app.route('/')
def index():
    # グローバル変数BASE_URLをテンプレートに渡す
    return render_template('index.html', base_url=BASE_URL)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f'File uploaded: {filename}')
        return jsonify({'filename': filename}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

def transcribe_job(job_id, filepath, txt_path):
    try:
        PROGRESS[job_id] = {'status': 'processing', 'progress': 0, 'message': '文字起こし中...'}
        CANCEL_FLAGS[job_id] = False
        logging.info(f"[JOB {job_id}] 文字起こし開始 ファイル: {filepath}")
        ext = os.path.splitext(filepath)[1].lower()
        if ext != '.wav':
            import ffmpeg
            wav_path = filepath + '.wav'
            (
                ffmpeg.input(filepath)
                .output(wav_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            audio_path = wav_path
            logging.info(f"[JOB {job_id}] ffmpegでwav変換完了: {wav_path}")
        else:
            audio_path = filepath

        total_duration = get_audio_duration(audio_path)
        start_time = time.time()
        log_filename = get_log_path()
        with open(log_filename, "a", encoding="utf-8") as logf:
            logf.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} ファイル: {audio_path}\n")

        segments, info = whisper_model.transcribe(audio_path, beam_size=5, language="ja")
        with open(txt_path, "w", encoding="utf-8") as f:
            last_end = 0
            for segment in segments:
                if CANCEL_FLAGS.get(job_id):
                    PROGRESS[job_id]['status'] = 'cancelled'
                    PROGRESS[job_id]['message'] = 'キャンセルされました'
                    logging.info(f'[JOB {job_id}] キャンセルされました')
                    print(f'\n[JOB {job_id}] キャンセルされました')
                    return
                f.write(segment.text + "\n")
                last_end = segment.end
                percent = min(100, int(last_end / total_duration * 100)) if total_duration else 0
                PROGRESS[job_id]['progress'] = percent
                PROGRESS[job_id]['message'] = f"文字起こし中...（{percent}%）"
                logging.info(f"[JOB {job_id}] 進捗: {percent}%")
                print(f"[JOB {job_id}] 進捗: {percent}%", end='\r', flush=True)
        PROGRESS[job_id]['progress'] = 100
        PROGRESS[job_id]['status'] = 'done'
        PROGRESS[job_id]['message'] = '完了'
        elapsed_min = (time.time() - start_time) / 60
        with open(log_filename, "a", encoding="utf-8") as logf:
            logf.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} 所要時間: {elapsed_min:.2f}分 ファイル: {audio_path}\n")
        logging.info(f'[JOB {job_id}] 文字起こし完了 所要時間: {elapsed_min:.2f}分 ファイル: {audio_path}')
        print(f'\n[JOB {job_id}] 文字起こし完了 所要時間: {elapsed_min:.2f}分 ファイル: {audio_path}')
    except Exception as e:
        PROGRESS[job_id]['status'] = 'error'
        PROGRESS[job_id]['message'] = 'エラーが発生しました'
        logging.error(f'[JOB {job_id}] Transcribe error: {e}')
        print(f'\n[JOB {job_id}] エラー: {e}')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'No filename'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    job_id = filename + '_' + datetime.now().strftime('%Y%m%d%H%M%S')
    txt_path = os.path.join(TRANSCRIBE_FOLDER, job_id + '.txt')
    thread = threading.Thread(target=transcribe_job, args=(job_id, filepath, txt_path))
    thread.start()
    logging.info(f'Transcribe started: {job_id}')
    return jsonify({'job_id': job_id}), 202

@app.route('/cancel/<job_id>', methods=['POST'])
def cancel(job_id):
    if job_id in PROGRESS and PROGRESS[job_id]['status'] == 'processing':
        CANCEL_FLAGS[job_id] = True
        return jsonify({'result': 'cancelling'}), 200
    return jsonify({'result': 'not_found_or_not_processing'}), 400

@app.route('/progress/<job_id>', methods=['GET'])
def progress(job_id):
    prog = PROGRESS.get(job_id)
    if not prog:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(prog)

@app.route('/download/<job_id>', methods=['GET'])
def download(job_id):
    txt_path = os.path.join(TRANSCRIBE_FOLDER, job_id + '.txt')
    if not os.path.exists(txt_path):
        return jsonify({'error': 'File not ready'}), 404
    return send_file(txt_path, as_attachment=True, download_name=job_id + '.txt')

if __name__ == '__main__':
    # ポート80で実行する場合は管理者権限が必要
    # Windowsの場合は管理者権限でコマンドプロンプトから実行する
    try:
        app.run(host='0.0.0.0', port=80, debug=False)
    except PermissionError:
        print('ポート80での実行に失敗しました。管理者権限が必要です。')
        print('代替としてポート5030で実行します...')
        app.run(host='0.0.0.0', port=5030, debug=False)
