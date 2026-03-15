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

@shared_task(bind=True)
def process_audio_file(self, audiofile_id):
    """
    Задача:
    - безопасно загрузить файл
    - попытаться использовать pydub для обработки (если доступен)
    - иначе использовать ffmpeg/ffprobe fallback
    - сохранить обработанный файл и duration
    """
    try:
        audio = AudioFile.objects.get(id=audiofile_id)
    except AudioFile.DoesNotExist:
        logger.error("AudioFile %s not found", audiofile_id)
        return {"status": "not_found"}

    file_field = audio.file
    if not file_field:
        logger.error("AudioFile %s has no file", audiofile_id)
        return {"status": "no_file"}

    # Считать файл в память
    try:
        # если локальный storage
        if hasattr(file_field, "path") and os.path.exists(file_field.path):
            with open(file_field.path, "rb") as f:
                raw = f.read()
        else:
            raw = file_field.read()
    except Exception as e:
        logger.exception("Failed to read file for %s: %s", audiofile_id, e)
        return {"status": "read_error", "error": str(e)}

    # Попытка использовать pydub (если установлен)
    try:
        from pydub import AudioSegment
        # pydub может определить формат по расширению
        try:
            sound = AudioSegment.from_file(io.BytesIO(raw))
            duration_sec = len(sound) / 1000.0
            # нормализация через pydub
            target_dBFS = -20.0
            change_in_dBFS = target_dBFS - sound.dBFS
            normalized = sound.apply_gain(change_in_dBFS)
            out_buf = io.BytesIO()
            normalized.export(out_buf, format="wav")
            out_buf.seek(0)
            processed_bytes = out_buf.read()
        except Exception as e:
            logger.exception("pydub processing failed, falling back to ffmpeg: %s", e)
            duration_sec = get_duration_with_ffprobe(raw)
            processed_bytes = normalize_with_ffmpeg(raw)
    except Exception as e:
        # pydub не доступен — используем ffmpeg fallback
        logger.info("pydub not available, using ffmpeg fallback: %s", e)
        try:
            duration_sec = get_duration_with_ffprobe(raw)
            processed_bytes = normalize_with_ffmpeg(raw)
        except Exception as e2:
            logger.exception("ffmpeg fallback failed for %s: %s", audiofile_id, e2)
            return {"status": "processing_error", "error": str(e2)}

    # Сохранить обработанный файл
    try:
        filename = os.path.basename(file_field.name)
        name_root, _ = os.path.splitext(filename)
        processed_name = f"{name_root}_processed_{int(timezone.now().timestamp())}.wav"
        audio.file.save(processed_name, ContentFile(processed_bytes), save=False)
        audio.duration = duration_sec
        audio.uploaded_at = timezone.now()
        audio.save()
    except Exception as e:
        logger.exception("Saving processed file failed for %s: %s", audiofile_id, e)
        return {"status": "save_error", "error": str(e)}

    logger.info("Processed audio %s duration %.2f s", audiofile_id, duration_sec)
    return {"status": "ok", "duration": duration_sec}
