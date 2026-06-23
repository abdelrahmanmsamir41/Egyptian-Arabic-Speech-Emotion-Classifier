import os
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from utils.file_utils import is_video_file


def convert_video_to_audio(video_path: str) -> str:
    """Extract a WAV audio track from a video file."""
    video = None
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise ValueError("Video file has no audio track")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = tmp.name
        video.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert video to audio: {e}") from e
    finally:
        if video is not None:
            try:
                if hasattr(video, "reader") and video.reader:
                    video.reader.close()
                if hasattr(video, "audio") and video.audio:
                    video.audio.reader.close_proc()
            except Exception:
                pass


def convert_to_wav(input_path: str) -> str:
    """Re-encode any audio file to WAV."""
    try:
        audio = AudioSegment.from_file(input_path)
        wav_path = input_path.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert audio to WAV: {e}") from e


def prepare_audio(path: str) -> str:
    """
    Given a raw video or audio path, return a path to a WAV file
    suitable for downstream processing.
    """
    if is_video_file(path):
        print("🎬 Extracting audio from video...")
        path = convert_video_to_audio(path)

    if not path.lower().endswith(".wav"):
        print("🎵 Converting to WAV...")
        path = convert_to_wav(path)

    return path
