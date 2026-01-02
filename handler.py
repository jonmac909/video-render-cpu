"""
RunPod Video Render Worker - CPU Version (32 vCPU optimized)
Renders videos from images with smoke + embers overlay effects using libx264.
"""

import runpod
import subprocess
import os
import tempfile
import requests
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=== Video Render Worker (CPU) Starting ===")

# Overlay files bundled with the worker
SMOKE_OVERLAY = "/app/overlays/smoke_gray.mp4"
EMBERS_OVERLAY = "/app/overlays/embers.mp4"

# FFmpeg settings - optimized for 32 vCPU
FFMPEG_THREADS = 32  # Use all available cores
FFMPEG_PRESET = "fast"  # Good speed/quality balance (options: ultrafast, superfast, veryfast, faster, fast, medium)
FFMPEG_CRF = "24"  # Constant quality (lower = better, 18-28 typical)

# Check overlay files exist
def check_overlays():
    """Verify overlay files are present."""
    smoke_ok = os.path.exists(SMOKE_OVERLAY)
    embers_ok = os.path.exists(EMBERS_OVERLAY)
    print(f"Overlay smoke_gray.mp4: {'✓' if smoke_ok else '✗'} {SMOKE_OVERLAY}")
    print(f"Overlay embers.mp4: {'✓' if embers_ok else '✗'} {EMBERS_OVERLAY}")
    if smoke_ok:
        smoke_size = os.path.getsize(SMOKE_OVERLAY)
        print(f"  smoke_gray.mp4 size: {smoke_size / 1024:.1f} KB")
    if embers_ok:
        embers_size = os.path.getsize(EMBERS_OVERLAY)
        print(f"  embers.mp4 size: {embers_size / 1024:.1f} KB")
    return smoke_ok and embers_ok

# Run startup checks
check_overlays()
print(f"Encoder: libx264 (CPU) with {FFMPEG_THREADS} threads")
print("=== Worker Ready ===")
print()


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', audio_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Failed to get audio duration: {e}")
    return 0.0


def parse_ffmpeg_progress(line: str, total_duration: float) -> float:
    """Parse FFmpeg progress line and return percentage (0-100)."""
    if total_duration <= 0:
        return 0.0
    match = re.search(r'time=(\d{2}):(\d{2}):(\d{2})\.(\d+)', line)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        ms = int(match.group(4)[:2]) / 100
        current_time = hours * 3600 + minutes * 60 + seconds + ms
        progress = (current_time / total_duration) * 100
        return min(progress, 100.0)
    return 0.0


def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """Download a file from URL to local path."""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        size = os.path.getsize(dest_path)
        print(f"Downloaded: {dest_path} ({size} bytes)")
        return True
    except Exception as e:
        print(f"Download failed: {url} - {e}")
        return False


def upload_to_supabase(file_path: str, bucket: str, storage_path: str,
                       supabase_url: str, supabase_key: str) -> str:
    """Upload file to Supabase storage using streaming."""
    try:
        file_size = os.path.getsize(file_path)
        print(f"Uploading to Supabase: {storage_path} ({file_size / 1024 / 1024:.1f} MB)")
        upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{storage_path}"
        headers = {
            'Authorization': f'Bearer {supabase_key}',
            'Content-Type': 'video/mp4',
            'x-upsert': 'true',
            'Content-Length': str(file_size)
        }
        with open(file_path, 'rb') as f:
            response = requests.post(upload_url, headers=headers, data=f, timeout=600)
        if response.status_code not in [200, 201]:
            print(f"Upload failed: {response.status_code} - {response.text}")
            raise Exception(f"Upload failed: {response.status_code}")
        public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
        print(f"Uploaded: {public_url}")
        return public_url
    except Exception as e:
        print(f"Upload error: {e}")
        raise


def update_render_job(supabase_url: str, supabase_key: str, job_id: str,
                      status: str, progress: int, message: str,
                      video_url: str = None, error: str = None) -> bool:
    """Update render job status in Supabase database."""
    try:
        if not job_id:
            return False
        print(f"Updating render job {job_id}: {status} ({progress}%)")
        update_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }
        if video_url:
            update_data["video_url"] = video_url
        if error:
            update_data["error"] = error
        url = f"{supabase_url}/rest/v1/render_jobs?id=eq.{job_id}"
        headers = {
            'Authorization': f'Bearer {supabase_key}',
            'apikey': supabase_key,
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }
        response = requests.patch(url, headers=headers, json=update_data, timeout=30)
        if response.status_code not in [200, 204]:
            print(f"Failed to update job status: {response.status_code} - {response.text}")
            return False
        return True
    except Exception as e:
        print(f"Error updating job status: {e}")
        return False


def create_concat_file(image_paths: list, timings: list) -> str:
    """Create FFmpeg concat demuxer file from images with timings."""
    concat_path = tempfile.mktemp(suffix='.txt')
    with open(concat_path, 'w') as f:
        for i, (img_path, timing) in enumerate(zip(image_paths, timings)):
            duration = timing['endSeconds'] - timing['startSeconds']
            f.write(f"file '{img_path}'\n")
            f.write(f"duration {duration}\n")
        if image_paths:
            f.write(f"file '{image_paths[-1]}'\n")
    return concat_path


def get_encoder_args():
    """Get libx264 encoder arguments optimized for 32 cores."""
    return [
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-threads', str(FFMPEG_THREADS)
    ]


def render_video_cpu(
    image_paths: list,
    timings: list,
    audio_path: str,
    output_path: str,
    apply_effects: bool = True,
    progress_callback=None
) -> bool:
    """
    Render video using CPU (libx264) with 32 threads.
    """
    encoder_args = get_encoder_args()
    total_duration = get_audio_duration(audio_path)
    print(f"Total audio duration: {total_duration:.1f}s")

    def send_progress(stage: str, percent: float, message: str):
        if progress_callback:
            progress_callback(stage, percent, message)
        print(f"[{stage}] {percent:.1f}% - {message}")

    try:
        concat_file = create_concat_file(image_paths, timings)

        if apply_effects:
            # Two-pass approach: render raw, then apply effects
            raw_video = tempfile.mktemp(suffix='_raw.mp4')

            cmd_raw = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=24',
                *encoder_args,
                '-pix_fmt', 'yuv420p',
                raw_video
            ]

            print(f"Pass 1: Rendering raw video (libx264, {FFMPEG_THREADS} threads)...")
            send_progress("rendering", 25, "Pass 1: Rendering raw video...")

            process = subprocess.Popen(
                cmd_raw,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            ffmpeg_output = []
            last_progress_time = time.time()
            last_progress_update = 0
            for line in process.stdout:
                ffmpeg_output.append(line)
                if 'frame=' in line or 'time=' in line or 'error' in line.lower():
                    print(f"  {line.strip()}")
                    last_progress_time = time.time()
                    if 'time=' in line and total_duration > 0:
                        ffmpeg_pct = parse_ffmpeg_progress(line, total_duration)
                        overall_pct = 25 + (ffmpeg_pct * 0.25)
                        now = time.time()
                        if now - last_progress_update >= 2:
                            last_progress_update = now
                            send_progress("rendering", overall_pct, f"Pass 1: {ffmpeg_pct:.0f}%")
                elif time.time() - last_progress_time > 30:
                    print(f"  Still rendering... (last: {line.strip()[:80]})")
                    last_progress_time = time.time()

            process.wait(timeout=7200)  # 2 hour timeout for CPU
            if process.returncode != 0:
                error_output = ''.join(ffmpeg_output[-20:])
                raise Exception(f"Raw render failed: {error_output[-500:]}")

            # Pass 2: Apply smoke + embers overlay
            cmd_effects = [
                'ffmpeg', '-y',
                '-i', raw_video,
                '-stream_loop', '-1', '-i', SMOKE_OVERLAY,
                '-stream_loop', '-1', '-i', EMBERS_OVERLAY,
                '-i', audio_path,
                '-filter_complex',
                '[1:v]colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3:0[smoke];'
                '[0:v][smoke]blend=all_mode=multiply[with_smoke];'
                '[2:v]colorkey=0x000000:0.2:0.2[embers];'
                '[with_smoke][embers]overlay=shortest=1[out]',
                '-map', '[out]',
                '-map', '3:a',
                *encoder_args,
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                output_path
            ]

            print(f"Pass 2: Applying effects (libx264, {FFMPEG_THREADS} threads)...")
            send_progress("rendering", 50, "Pass 2: Applying smoke + embers...")

            process = subprocess.Popen(
                cmd_effects,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            ffmpeg_output = []
            last_progress_time = time.time()
            last_progress_update = 0
            for line in process.stdout:
                ffmpeg_output.append(line)
                if 'frame=' in line or 'time=' in line or 'error' in line.lower():
                    print(f"  {line.strip()}")
                    last_progress_time = time.time()
                    if 'time=' in line and total_duration > 0:
                        ffmpeg_pct = parse_ffmpeg_progress(line, total_duration)
                        overall_pct = 50 + (ffmpeg_pct * 0.35)
                        now = time.time()
                        if now - last_progress_update >= 2:
                            last_progress_update = now
                            send_progress("rendering", overall_pct, f"Pass 2: {ffmpeg_pct:.0f}%")
                elif time.time() - last_progress_time > 30:
                    print(f"  Still rendering... (last: {line.strip()[:80]})")
                    last_progress_time = time.time()

            process.wait(timeout=7200)

            if os.path.exists(raw_video):
                os.remove(raw_video)

            if process.returncode != 0:
                error_output = ''.join(ffmpeg_output[-20:])
                raise Exception(f"Effects render failed: {error_output[-500:]}")

        else:
            # No effects - single pass with audio
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-i', audio_path,
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=24',
                *encoder_args,
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                output_path
            ]

            print(f"Rendering video without effects (libx264, {FFMPEG_THREADS} threads)...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode != 0:
                raise Exception(f"Render failed: {result.stderr[-500:]}")

        if os.path.exists(concat_file):
            os.remove(concat_file)

        output_size = os.path.getsize(output_path)
        print(f"Render complete: {output_path} ({output_size / 1024 / 1024:.1f} MB)")
        return True

    except subprocess.TimeoutExpired:
        raise Exception("Render timed out after 2 hours")
    except Exception as e:
        print(f"Render error: {e}")
        raise


def handler(job):
    """RunPod handler for video rendering (CPU version)."""
    job_input = job["input"]

    image_urls = job_input.get("image_urls", [])
    timings = job_input.get("timings", [])
    audio_url = job_input.get("audio_url")
    project_id = job_input.get("project_id")
    apply_effects = job_input.get("apply_effects", True)
    supabase_url = job_input.get("supabase_url")
    supabase_key = job_input.get("supabase_key")
    render_job_id = job_input.get("render_job_id")

    if not image_urls:
        return {"error": "No image URLs provided"}
    if not audio_url:
        return {"error": "No audio URL provided"}
    if not project_id:
        return {"error": "No project ID provided"}
    if not supabase_url or not supabase_key:
        return {"error": "Supabase credentials required"}
    if len(image_urls) != len(timings):
        return {"error": "Image URLs and timings count mismatch"}

    if apply_effects:
        if not os.path.exists(SMOKE_OVERLAY) or not os.path.exists(EMBERS_OVERLAY):
            return {"error": "Overlay files missing - cannot apply effects"}

    print(f"Starting video render: {len(image_urls)} images, effects={apply_effects}, encoder=libx264 ({FFMPEG_THREADS} threads)")
    start_time = time.time()

    last_supabase_update = [0]
    def progress_callback(stage: str, percent: float, message: str):
        now = time.time()
        if now - last_supabase_update[0] >= 2:
            last_supabase_update[0] = now
            update_render_job(
                supabase_url, supabase_key, render_job_id,
                status="rendering",
                progress=int(percent),
                message=message
            )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            update_render_job(supabase_url, supabase_key, render_job_id,
                              status="downloading", progress=5, message="Downloading assets...")

            download_start = time.time()
            image_paths = [None] * len(image_urls)
            failed_downloads = []

            def download_image(args):
                i, url = args
                ext = '.png' if '.png' in url.lower() else '.jpg'
                img_path = os.path.join(temp_dir, f"image_{i:03d}{ext}")
                success = download_file(url, img_path)
                return i, img_path, success

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(download_image, (i, url)): i
                          for i, url in enumerate(image_urls)}
                completed_count = 0
                for future in as_completed(futures):
                    i, img_path, success = future.result()
                    if success:
                        image_paths[i] = img_path
                    else:
                        failed_downloads.append(i)
                    completed_count += 1
                    if completed_count % 10 == 0:
                        dl_pct = 5 + int((completed_count / len(image_urls)) * 15)
                        update_render_job(supabase_url, supabase_key, render_job_id,
                                          status="downloading", progress=dl_pct,
                                          message=f"Downloaded {completed_count}/{len(image_urls)} images")

            if failed_downloads:
                return {"error": f"Failed to download images: {failed_downloads[:5]}"}

            download_elapsed = time.time() - download_start
            print(f"Downloaded {len(image_paths)} images in {download_elapsed:.1f}s")

            update_render_job(supabase_url, supabase_key, render_job_id,
                              status="downloading", progress=22, message="Downloading audio...")
            audio_path = os.path.join(temp_dir, "audio.wav")
            if not download_file(audio_url, audio_path, timeout=300):
                return {"error": "Failed to download audio"}

            update_render_job(supabase_url, supabase_key, render_job_id,
                              status="rendering", progress=25, message="Starting video render...")
            output_path = os.path.join(temp_dir, "output.mp4")
            render_video_cpu(image_paths, timings, audio_path, output_path, apply_effects,
                            progress_callback=progress_callback)

            update_render_job(supabase_url, supabase_key, render_job_id,
                              status="uploading", progress=88, message="Uploading video...")
            storage_path = f"{project_id}/video.mp4"
            video_url = upload_to_supabase(
                output_path,
                "generated-assets",
                storage_path,
                supabase_url,
                supabase_key
            )

            elapsed = time.time() - start_time
            print(f"Total time: {elapsed:.1f}s")

            update_render_job(
                supabase_url, supabase_key, render_job_id,
                status="complete",
                progress=100,
                message=f"Video rendered successfully (CPU: {elapsed:.1f}s)",
                video_url=video_url
            )

            return {
                "video_url": video_url,
                "render_time_seconds": elapsed
            }

    except Exception as e:
        print(f"Handler error: {e}")
        update_render_job(
            supabase_url, supabase_key, render_job_id,
            status="failed",
            progress=0,
            message="CPU render failed",
            error=str(e)
        )
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
