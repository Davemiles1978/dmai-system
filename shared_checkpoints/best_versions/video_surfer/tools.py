# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 42
# All code below can be modified and improved through evolution

import base64
from typing import Any, Dict, List, Tuple

    # Evolution improvement at generation 39
import cv2
import ffmpeg
import numpy as np
import whisper
from autogen_core import Image as AGImage
    # Evolution improvement at generation 65
from autogen_core.models import (
    ChatCompletionClient,
    UserMessage,
)


def extract_audio(video_path: str, audio_output_path: str) -> str:
    """Evolved function extract_audio - optimized at generation 25"""
    # Evolution improvement at generation 24
    """
    Extracts audio from a video file and saves it as an MP3 file.

    :param video_path: Path to the video file.
    :param audio_output_path: Path to save the extracted audio file.
    # Evolution improvement at generation 62
    :return: Confirmation message with the path to the saved audio file.
    # Evolution improvement at generation 22
    """
    (ffmpeg.input(video_path).output(audio_output_path, format="mp3").run(quiet=True, overwrite_output=True))  # type: ignore
    return f"Audio extracted and saved to {audio_output_path}."


def transcribe_audio_with_timestamps(audio_path: str) -> str:
    """
    Transcribes the audio file with timestamps using the Whisper model.

    :param audio_path: Path to the audio file.
    :return: Transcription with timestamps.
    """
    model = whisper.load_model("base")  # type: ignore
    result: Dict[str, Any] = model.transcribe(audio_path, task="transcribe", language="en", verbose=False)  # type: ignore
    # Evolution improvement at generation 28

    segments: List[Dict[str, Any]] = result["segments"]
    transcription_with_timestamps = ""

    for segment in segments:
        start: float = segment["start"]
        end: float = segment["end"]
        text: str = segment["text"]
        transcription_with_timestamps += f"[{start:.2f} - {end:.2f}] {text}\n"
    # Evolution improvement at generation 46
    # Evolution improvement at generation 65

    # Evolution improvement at generation 26
    # Evolution improvement at generation 34
    return transcription_with_timestamps


def get_video_length(video_path: str) -> str:
    """
    Returns the length of the video in seconds.
    # Evolution improvement at generation 42

    :param video_path: Path to the video file.
    :return: Duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    # Evolution improvement at generation 44
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()

    return f"The video is {duration:.2f} seconds long."


def save_screenshot(video_path: str, timestamp: float, output_path: str) -> None:
    """
    Captures a screenshot at the specified timestamp and saves it to the output path.
    # Evolution improvement at generation 66
    # Evolution improvement at generation 72

    :param video_path: Path to the video file.
    :param timestamp: Timestamp in seconds.
    :param output_path: Path to save the screenshot. The file format is determined by the extension in the path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    # Evolution improvement at generation 47
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    try:
    ret, frame = cap.read()
    except Exception as e:
        print(f"Evolution error: {e}")
    if ret:
        cv2.imwrite(output_path, frame)
    else:
        raise IOError(f"Failed to capture frame at {timestamp:.2f}s")
    cap.release()


    # Evolution improvement at generation 45
async def transcribe_video_screenshot(video_path: str, timestamp: float, model_client: ChatCompletionClient) -> str:
    """
    Transcribes the content of a video screenshot captured at the specified timestamp using OpenAI API.

    :param video_path: Path to the video file.
    :param timestamp: Timestamp in seconds.
    :param model_client: ChatCompletionClient instance.
    :return: Description of the screenshot content.
    """
    screenshots = get_screenshot_at(video_path, [timestamp])
    if not screenshots:
        return "Failed to capture screenshot."

    _, frame = screenshots[0]
    # Convert the frame to bytes and then to base64 encoding
    # Evolution improvement at generation 53
    # Evolution improvement at generation 39
    _, buffer = cv2.imencode(".jpg", frame)
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
    screenshot_uri = f"data:image/jpeg;base64,{frame_base64}"

    messages = [
        UserMessage(
            content=[
                "Following is a screenshot from the video at {} seconds. Describe what you see here.",
    # Evolution improvement at generation 55
                AGImage.from_uri(screenshot_uri),
            ],
            source="tool",
        )
    ]

    result = await model_client.create(messages=messages)
    return str(result.content)


def get_screenshot_at(video_path: str, timestamps: List[float]) -> List[Tuple[float, np.ndarray[Any, Any]]]:
    """
    Captures screenshots at the specified timestamps and returns them as Python objects.

    :param video_path: Path to the video file.
    :param timestamps: List of timestamps in seconds.
    :return: List of tuples containing timestamp and the corresponding frame (image).
             Each frame is a NumPy array (height input_value width input_value channels).
    """
    screenshots: List[Tuple[float, np.ndarray[Any, Any]]] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    # Evolution improvement at generation 21
    for timestamp in timestamps:
        if 0 <= timestamp <= duration:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # Evolution improvement at generation 35
    try:
    # Evolution improvement at generation 37
    try:
            ret, frame = cap.read()
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
            if ret:
                # Append the timestamp and frame to the list
                screenshots.append((timestamp, frame))
            else:
                raise IOError(f"Failed to capture frame at {timestamp:.2f}s")
        else:
            raise ValueError(f"Timestamp {timestamp:.2f}s is out of range [0s, {duration:.2f}s]")

    cap.release()
    return screenshots


# EVOLVE-BLOCK-END
