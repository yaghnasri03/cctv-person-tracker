from fastapi import APIRouter, UploadFile, File
import os
import shutil
import uuid
import subprocess
from pathlib import Path

from video_processor import PersonDetector

router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = PersonDetector()


# Exact FFmpeg location on this computer
FFMPEG_BIN = "ffmpeg"


def convert_to_browser_mp4(
    input_path: Path,
    output_path: Path
) -> None:

    """
    Convert the processed video into a browser-compatible MP4
    using H.264 video and AAC audio.
    """

    command = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    print("Running FFmpeg conversion...")
    print("FFmpeg path:", FFMPEG_BIN)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:

        print("FFmpeg ERROR:")
        print(result.stderr)

        raise RuntimeError(
            "FFmpeg conversion failed:\n"
            + result.stderr[-4000:]
        )

    print(
        "FFmpeg conversion completed successfully."
    )


@router.post("/upload")
async def upload_files(
    video: UploadFile = File(...),
    photo: UploadFile = File(...)
):

    print("===== Upload API Called =====")

    job_id = str(uuid.uuid4())

    job_folder = os.path.join(
        UPLOAD_DIR,
        job_id
    )

    os.makedirs(
        job_folder,
        exist_ok=True
    )

    video_path = os.path.join(
        job_folder,
        video.filename
    )

    photo_path = os.path.join(
        job_folder,
        photo.filename
    )

    print("Saving files...")

    with open(video_path, "wb") as buffer:

        shutil.copyfileobj(
            video.file,
            buffer
        )

    with open(photo_path, "wb") as buffer:

        shutil.copyfileobj(
            photo.file,
            buffer
        )

    print("Files saved.")

    # --------------------------------------------------
    # Processed video created by OpenCV
    # --------------------------------------------------

    output_filename = (
        f"{job_id}_processed.mp4"
    )

    output_video_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    print("Starting detection...")

    detection_result = (
        detector.process_video_for_detection(
            video_path,
            output_video_path,
            photo_path
        )
    )

    print("Detection finished.")

    # --------------------------------------------------
    # Extract detection results
    # --------------------------------------------------

    person_found = detection_result.get(
        "person_found",
        False
    )

    match_count = detection_result.get(
        "match_count",
        0
    )

    timestamps = detection_result.get(
        "timestamps",
        []
    )

    print(
        "Person found:",
        person_found
    )

    print(
        "Match count:",
        match_count
    )

    print(
        "Timestamps:",
        timestamps
    )

    # --------------------------------------------------
    # Convert video for browser
    # --------------------------------------------------

    web_output_filename = (
        f"{job_id}_web.mp4"
    )

    web_output_video_path = os.path.join(
        OUTPUT_DIR,
        web_output_filename
    )

    print(
        "Converting video for browser playback..."
    )

    convert_to_browser_mp4(
        Path(output_video_path),
        Path(web_output_video_path)
    )

    print(
        "Browser-compatible video created."
    )

    # --------------------------------------------------
    # Send everything back to React
    # --------------------------------------------------

    return {

        "status": "success",

        "job_id": job_id,

        "video": video.filename,

        "photo": photo.filename,

        "output_video": (
            f"/outputs/{output_filename}"
        ),

        "video_url": (
            f"http://127.0.0.1:8000/"
            f"outputs/{web_output_filename}"
        ),

        "person_found": person_found,

        "match_count": match_count,

        "timestamps": timestamps,
    }