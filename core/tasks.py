# core/tasks.py
import os
import io
import subprocess
import logging
from celery import shared_task
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

from .models import AudioFile

logger = logging.getLogger(__name__)

def get_duration_with_ffprobe(file_bytes):
    """
    Возвращает длительность в секундах, используя ffprobe (читает из stdin через временный файл).
    """
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", tmp_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        duration = float(out.strip())
        return duration
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def normalize_with_ffmpeg(file_bytes):
    """
    Нормализует аудио до -20 dBFS с помощью ffmpeg и возвращает bytes WAV.
    Использует loudnorm фильтр (two-pass) — простой вариант.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as in_tmp:
        in_tmp.write(file_bytes)
        in_tmp.flush()
        in_path = in_tmp.name

    out_path = in_path + "_norm.wav"
    try:
        # Простой one-pass gain normalization через loudnorm (можно улучшить)
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
            "-ar", "44100", "-ac", "2",
            out_path
        ]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        with open(out_path, "rb") as f:
            data = f.read()
        return data
    finally:
        for p in (in_path, out_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

try:
    import noisereduce as nr
    HAS_NR = True
except Exception:
    HAS_NR = False

@shared_task(bind=True)
def process_audio_file(self, audiofile_id, mode='denoise'):
    af = AudioFile.objects.get(id=audiofile_id)
    af.status = 'processing'
    af.save(update_fields=['status'])

    src_path = af.file.path
    tmp_wav = src_path + ".tmp.wav"

    # Конвертируем в WAV 16k mono
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        tmp_wav
    ]
    subprocess.run(cmd, check=True)

    data, sr = sf.read(tmp_wav)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if mode == 'denoise' and HAS_NR:
        reduced = nr.reduce_noise(y=data, sr=sr)
    else:
        reduced = data

    out_name = f"processed_{af.id}.wav"
    out_wav = os.path.join(settings.MEDIA_ROOT, out_name)
    sf.write(out_wav, reduced, sr, subtype='PCM_16')

    af.file.name = out_name
    af.duration = len(reduced) / sr
    af.status = 'done'
    af.save(update_fields=['file', 'duration', 'status'])

    try:
        os.remove(tmp_wav)
    except OSError:
        pass

    return {'status': 'ok', 'duration': af.duration}
