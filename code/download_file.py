import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
SPLITS = ("train", "val", "test")
CLIP_DIR = DATA_DIR / "source_clips"
IMAGE_DIR = DATA_DIR / "images"


def load_segments(spatial):
    sources = defaultdict(list)
    for split in SPLITS:
        with (DATA_DIR / "opencv" / f"hapact_{split}_opencv.json").open(encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            segment_id = record["segment_id"]
            prefix, start, end, _ = segment_id.rsplit("_", 3)
            youtube_id = prefix.removeprefix(record["imdb_id"] + "_")
            frames = set()
            if spatial:
                for entry in record["smplx_vertices"]:
                    for event in entry.values():
                        frames.update(range(event["frames"][0], event["frames"][1] + 1))
            sources[youtube_id].append((segment_id, start, end, record["total_frames"], frames))
    return sources


def valid_clip(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=avg_frame_rate,nb_read_frames", "-of", "json", str(path)
        ], capture_output=True, text=True, check=True)
        stream = json.loads(result.stdout)["streams"][0]
        return Fraction(stream["avg_frame_rate"]) == 24 and int(stream["nb_read_frames"]) > 0
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return False


def download_source(youtube_id, temp_dir, browser=None):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    output = temp_dir / "%(id)s.%(ext)s"
    command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f", "bv*+ba/b", "-o", str(output)]
    if browser:
        command.extend(["--cookies-from-browser", browser])
    subprocess.run(command + [url], check=True)
    originals = [p for p in temp_dir.iterdir() if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
    if not originals:
        raise FileNotFoundError(f"Downloaded video not found: {youtube_id}")
    return originals[0]


def create_clip(original, segment):
    segment_id, start, end, total_frames, _ = segment
    output = CLIP_DIR / f"{segment_id}.mp4"
    partial = CLIP_DIR / f"{segment_id}.partial.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", start, "-i", str(original), "-t", str(float(end) - float(start)),
            "-map", "0:v:0", "-map", "0:a:0?", "-vf", "fps=24",
            "-frames:v", str(total_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(partial)
        ], check=True)
        if not valid_clip(partial):
            raise RuntimeError(f"Invalid clip (expected 24 FPS with readable frames): {segment_id}")
        partial.replace(output)
        print(f"Saved {output}")
    finally:
        partial.unlink(missing_ok=True)


def extract_frames(segment):
    import cv2

    segment_id, _, _, _, frames = segment
    pending = set()
    for frame in frames:
        path = IMAGE_DIR / f"{segment_id}_{frame:06d}.png"
        if not path.exists() or cv2.imread(str(path)) is None:
            pending.add(frame)
    if not pending:
        return

    video = cv2.VideoCapture(str(CLIP_DIR / f"{segment_id}.mp4"))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open clip: {segment_id}")
    try:
        for index in range(max(pending) + 1):
            ok, image = video.read()
            if not ok:
                raise RuntimeError(f"Missing frame {index}: {segment_id}")
            if index in pending:
                path = IMAGE_DIR / f"{segment_id}_{index:06d}.png"
                partial = IMAGE_DIR / f"{segment_id}_{index:06d}.partial.png"
                try:
                    if not cv2.imwrite(str(partial), image) or cv2.imread(str(partial)) is None:
                        raise RuntimeError(f"Cannot save image: {path}")
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
        print(f"Extracted {len(pending)} frames: {segment_id}")
    finally:
        video.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--cookies-from-browser")
    args = parser.parse_args()

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    if args.spatial:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_segments(args.spatial)
    success = failed = skipped = 0
    attempted = False
    for youtube_id, segments in sources.items():
        missing = [segment for segment in segments if not valid_clip(CLIP_DIR / f"{segment[0]}.mp4")]
        skipped += len(segments) - len(missing)
        if missing:
            if attempted:
                time.sleep(1)
            attempted = True
            try:
                with tempfile.TemporaryDirectory() as directory:
                    original = download_source(youtube_id, Path(directory), args.cookies_from_browser)
                    for segment in missing:
                        try:
                            create_clip(original, segment)
                            success += 1
                        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                            failed += 1
                            print(f"Failed to create {segment[0]}: {exc}")
            except (OSError, subprocess.CalledProcessError) as exc:
                failed += len(missing)
                print(f"Failed to download {youtube_id}: {exc}")

    print(f"Download complete: success={success}, failed={failed}, skipped={skipped}")

    if args.spatial:
        for segments in sources.values():
            for segment in segments:
                if valid_clip(CLIP_DIR / f"{segment[0]}.mp4"):
                    try:
                        extract_frames(segment)
                    except (OSError, RuntimeError) as exc:
                        print(f"Failed to extract {segment[0]}: {exc}")


if __name__ == "__main__":
    main()
