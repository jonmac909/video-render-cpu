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
# smoke_gray_loop.mp4 is a ping-pong (forward+reverse) version for seamless looping
SMOKE_OVERLAY = "/app/overlays/smoke_gray_loop.mp4"
EMBERS_OVERLAY = "/app/overlays/embers.mp4"

# FFmpeg settings - optimized for 32 vCPU with 64GB RAM
FFMPEG_THREADS = 0  # 0 = auto-detect optimal thread count (recommended over hardcoding)
FFMPEG_PRESET = "veryfast"  # Faster encoding with minimal quality loss (veryfast is ~2x faster than fast)
FFMPEG_CRF = "23"  # Slightly better quality (23 is x264 default, visually lossless)
FFMPEG_LOOKAHEAD = 50  # Frames to look ahead for better encoding decisions
FFMPEG_THREAD_TYPE = "frame"  # Frame-level parallelism (best for many cores)

# Check overlay files exist
def check_overlays():
    """Verify overlay files are present."""
    smoke_ok = os.path.exists(SMOKE_OVERLAY)
    embers_ok = os.path.exists(EMBERS_OVERLAY)
    print(f"Overlay smoke_gray_loop.mp4: {'✓' if smoke_ok else '✗'} {SMOKE_OVERLAY}")
    print(f"Overlay embers.mp4: {'✓' if embers_ok else '✗'} {EMBERS_OVERLAY}")
    if smoke_ok:
        smoke_size = os.path.getsize(SMOKE_OVERLAY)
        print(f"  smoke_gray_loop.mp4 size: {smoke_size / 1024:.1f} KB")
    if embers_ok:
        embers_size = os.path.getsize(EMBERS_OVERLAY)
        print(f"  embers.mp4 size: {embers_size / 1024:.1f} KB")
    return smoke_ok and embers_ok

# Run startup checks
check_overlays()
cpu_count = os.cpu_count() or 32
print(f"Encoder: libx264 (CPU) with auto threads ({cpu_count} cores detected)")
print(f"Preset: {FFMPEG_PRESET}, CRF: {FFMPEG_CRF}, Thread type: {FFMPEG_THREAD_TYPE}")
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
    """Get libx264 encoder arguments optimized for 32 cores with 64GB RAM."""
    return [
        '-c:v', 'libx264',
        '-preset', FFMPEG_PRESET,
        '-crf', FFMPEG_CRF,
        '-threads', str(FFMPEG_THREADS),  # 0 = auto-detect
        '-x264-params', f'threads=auto:lookahead_threads=auto:rc-lookahead={FFMPEG_LOOKAHEAD}',
        '-thread_type', FFMPEG_THREAD_TYPE,  # frame-level parallelism
    ]


def get_chunk_duration(timings: list) -> float:
    """Calculate total duration from timings (for chunk mode without audio)."""
    if not timings:
        return 0.0
    return sum(t['endSeconds'] - t['startSeconds'] for t in timings)


def render_video_cpu(
    image_paths: list,
    timings: list,
    audio_path: str,
    output_path: str,
    apply_effects: bool = True,
    progress_callback=None,
    chunk_mode: bool = False
) -> bool:
    """
    Render video using CPU (libx264) with 32 threads.
    In chunk_mode, renders video-only (no audio) for later concatenation.
    """
    encoder_args = get_encoder_args()

    # In chunk mode, calculate duration from timings (no audio file)
    if chunk_mode:
        total_duration = get_chunk_duration(timings)
        print(f"Chunk duration (from timings): {total_duration:.1f}s")
    else:
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

            cpu_cores = os.cpu_count() or 32
            cmd_raw = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-filter_threads', str(cpu_cores),  # Parallelize filter operations
                '-filter_complex_threads', str(cpu_cores),  # For complex filter graphs
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=24',
                *encoder_args,
                '-pix_fmt', 'yuv420p',
                raw_video
            ]

            print(f"Pass 1: Rendering raw video (libx264, {cpu_cores} cores)...")
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
            # Note: flags=full_chroma_int+accurate_rnd prevents green tint from color conversion
            # Grayscale coefficients: .3:.59:.11 (standard luminance weights for R, G, B)
            if chunk_mode:
                # Chunk mode: video only, no audio
                # CRITICAL: Use shortest=1 on blend filter to prevent infinite looping
                # (smoke/embers overlays are -stream_loop -1, so blend must limit to base video duration)
                cmd_effects = [
                    'ffmpeg', '-y',
                    '-i', raw_video,
                    '-stream_loop', '-1', '-i', SMOKE_OVERLAY,
                    '-stream_loop', '-1', '-i', EMBERS_OVERLAY,
                    '-filter_threads', str(cpu_cores),
                    '-filter_complex_threads', str(cpu_cores),
                    '-filter_complex',
                    '[0:v]scale=1920:1080:flags=full_chroma_int+accurate_rnd[base];'
                    '[1:v]scale=1920:1080:flags=full_chroma_int+accurate_rnd,colorchannelmixer=.3:.59:.11:0:.3:.59:.11:0:.3:.59:.11:0[smoke_gray];'
                    '[base][smoke_gray]blend=all_mode=multiply:shortest=1[with_smoke];'
                    '[2:v]scale=1920:1080,colorkey=black:similarity=0.2:blend=0.2[embers_keyed];'
                    '[with_smoke][embers_keyed]overlay=0:0:shortest=1[out]',
                    '-map', '[out]',
                    *encoder_args,
                    '-pix_fmt', 'yuv420p',
                    output_path
                ]
            else:
                # Full mode: video + audio
                cmd_effects = [
                    'ffmpeg', '-y',
                    '-i', raw_video,
                    '-stream_loop', '-1', '-i', SMOKE_OVERLAY,
                    '-stream_loop', '-1', '-i', EMBERS_OVERLAY,
                    '-i', audio_path,
                    '-filter_threads', str(cpu_cores),
                    '-filter_complex_threads', str(cpu_cores),
                    '-filter_complex',
                    '[0:v]scale=1920:1080:flags=full_chroma_int+accurate_rnd[base];'
                    '[1:v]scale=1920:1080:flags=full_chroma_int+accurate_rnd,colorchannelmixer=.3:.59:.11:0:.3:.59:.11:0:.3:.59:.11:0[smoke_gray];'
                    '[base][smoke_gray]blend=all_mode=multiply[with_smoke];'
                    '[2:v]scale=1920:1080,colorkey=black:similarity=0.2:blend=0.2[embers_keyed];'
                    '[with_smoke][embers_keyed]overlay=0:0:shortest=1[out]',
                    '-map', '[out]',
                    '-map', '3:a',
                    *encoder_args,
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    output_path
                ]

            print(f"Pass 2: Applying effects (libx264, {cpu_cores} cores)...")
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
            # No effects - single pass
            cpu_cores = os.cpu_count() or 32
            if chunk_mode:
                # Chunk mode: video only, no audio
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-filter_threads', str(cpu_cores),
                    '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=24',
                    *encoder_args,
                    '-pix_fmt', 'yuv420p',
                    output_path
                ]
            else:
                # Full mode: video + audio
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-i', audio_path,
                    '-filter_threads', str(cpu_cores),
                    '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=24',
                    *encoder_args,
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    output_path
                ]

            print(f"Rendering video without effects (libx264, {cpu_cores} cores)...")
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


def finalize_video(
    chunk_urls: list,
    audio_url: str,
    output_path: str,
    temp_dir: str,
    progress_callback=None
) -> bool:
    """
    Finalize video by concatenating chunks and muxing audio.
    This runs on RunPod to leverage 32 cores for audio encoding.
    """
    cpu_cores = os.cpu_count() or 32

    def send_progress(percent: float, message: str):
        if progress_callback:
            progress_callback("finalizing", percent, message)
        print(f"[finalize] {percent:.0f}% - {message}")

    send_progress(5, "Downloading video chunks...")

    # Download all chunks in parallel
    chunk_paths = []
    failed_chunks = []

    def download_chunk(args):
        i, url = args
        chunk_path = os.path.join(temp_dir, f"chunk_{i:02d}.mp4")
        success = download_file(url, chunk_path, timeout=300)
        return i, chunk_path, success

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_chunk, (i, url)): i
                   for i, url in enumerate(chunk_urls)}
        for future in as_completed(futures):
            i, chunk_path, success = future.result()
            if success:
                chunk_paths.append((i, chunk_path))
            else:
                failed_chunks.append(i)

    if failed_chunks:
        raise Exception(f"Failed to download chunks: {failed_chunks}")

    # Sort by index to maintain order
    chunk_paths.sort(key=lambda x: x[0])
    chunk_paths = [p for _, p in chunk_paths]

    send_progress(25, "Downloading audio...")

    # Download audio WAV
    audio_path = os.path.join(temp_dir, "audio.wav")
    if not download_file(audio_url, audio_path, timeout=600):
        raise Exception("Failed to download audio")

    audio_duration = get_audio_duration(audio_path)
    print(f"Audio duration: {audio_duration:.1f}s")

    send_progress(35, f"Encoding audio with {cpu_cores} cores...")

    # Encode WAV to AAC (fast with 32 cores)
    audio_aac_path = os.path.join(temp_dir, "audio.m4a")
    encode_cmd = [
        'ffmpeg', '-y',
        '-i', audio_path,
        '-c:a', 'aac',
        '-ar', '48000',
        '-b:a', '192k',
        '-threads', '0',
        audio_aac_path
    ]

    print(f"Encoding audio: {' '.join(encode_cmd[:8])}...")
    process = subprocess.Popen(
        encode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    last_pct = 0
    for line in process.stdout:
        if 'time=' in line and audio_duration > 0:
            pct = parse_ffmpeg_progress(line, audio_duration)
            if pct > last_pct + 10:
                last_pct = pct
                send_progress(35 + pct * 0.15, f"Encoding audio... {pct:.0f}%")

    process.wait(timeout=1800)  # 30 min timeout for audio
    if process.returncode != 0:
        raise Exception("Audio encoding failed")

    print("Audio encoded to AAC")

    send_progress(50, "Concatenating video chunks...")

    # Create concat file
    concat_file = os.path.join(temp_dir, "chunks.txt")
    with open(concat_file, 'w') as f:
        for chunk_path in chunk_paths:
            f.write(f"file '{chunk_path}'\n")

    # Concatenate chunks (stream copy - instant)
    concat_path = os.path.join(temp_dir, "concatenated.mp4")
    concat_cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        concat_path
    ]

    print("Concatenating chunks (stream copy)...")
    result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise Exception(f"Concat failed: {result.stderr[-500:]}")

    send_progress(60, "Muxing audio with video...")

    # Mux audio with video (stream copy - instant)
    mux_cmd = [
        'ffmpeg', '-y',
        '-i', concat_path,
        '-i', audio_aac_path,
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-shortest',
        '-movflags', '+faststart',
        output_path
    ]

    print("Muxing audio (stream copy)...")
    result = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise Exception(f"Mux failed: {result.stderr[-500:]}")

    # Cleanup intermediate files
    for chunk_path in chunk_paths:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(audio_aac_path):
        os.remove(audio_aac_path)
    if os.path.exists(concat_path):
        os.remove(concat_path)

    output_size = os.path.getsize(output_path)
    print(f"Finalize complete: {output_path} ({output_size / 1024 / 1024:.1f} MB)")
    return True


def handler(job):
    """RunPod handler for video rendering (CPU version)."""
    print(f"=== Handler invoked ===")
    print(f"Job keys: {list(job.keys())}")
    try:
        job_input = job.get("input", {})
        print(f"Input keys: {list(job_input.keys())}")
        print(f"chunk_mode: {job_input.get('chunk_mode')}")
        print(f"finalize_mode: {job_input.get('finalize_mode')}")
        print(f"image_urls count: {len(job_input.get('image_urls', []))}")
    except Exception as e:
        print(f"Input parsing error: {e}")
        return {"error": f"Input parsing failed: {str(e)}"}

    image_urls = job_input.get("image_urls", [])
    timings = job_input.get("timings", [])
    audio_url = job_input.get("audio_url")
    project_id = job_input.get("project_id")
    apply_effects = job_input.get("apply_effects", True)
    supabase_url = job_input.get("supabase_url")
    supabase_key = job_input.get("supabase_key")
    render_job_id = job_input.get("render_job_id")

    # Chunk mode parameters for parallel rendering
    chunk_mode = job_input.get("chunk_mode", False)
    chunk_index = job_input.get("chunk_index", 0)
    total_chunks = job_input.get("total_chunks", 1)

    # Finalize mode: concatenate chunks + encode audio + mux (all on RunPod)
    finalize_mode = job_input.get("finalize_mode", False)
    chunk_urls = job_input.get("chunk_urls", [])

    # Handle finalize mode separately (different workflow)
    if finalize_mode:
        print(f"FINALIZE MODE: chunk_urls={len(chunk_urls)}, audio_url={bool(audio_url)}, project_id={project_id}")
        print(f"FINALIZE MODE: supabase_url={bool(supabase_url)}, supabase_key={bool(supabase_key)}, render_job_id={render_job_id}")
        if not chunk_urls:
            return {"error": "No chunk URLs provided for finalize mode"}
        if not audio_url:
            return {"error": "No audio URL provided for finalize mode"}
        if not project_id:
            return {"error": "No project ID provided"}
        if not supabase_url or not supabase_key:
            return {"error": "Supabase credentials required"}
        if not render_job_id:
            return {"error": "No render_job_id provided for finalize mode"}

        cpu_cores = os.cpu_count() or 32
        print(f"Starting FINALIZE mode: {len(chunk_urls)} chunks, audio encoding with {cpu_cores} cores")
        start_time = time.time()

        last_supabase_update = [0]
        def finalize_progress_callback(stage: str, percent: float, message: str):
            now = time.time()
            if now - last_supabase_update[0] >= 2:
                last_supabase_update[0] = now
                # Map finalize progress to 76-95% range (where muxing phase lives)
                mapped_pct = 76 + int(percent * 0.19)
                update_render_job(
                    supabase_url, supabase_key, render_job_id,
                    status="muxing",
                    progress=mapped_pct,
                    message=message
                )

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                update_render_job(supabase_url, supabase_key, render_job_id,
                                  status="muxing", progress=76, message="Finalizing: downloading chunks...")

                output_path = os.path.join(temp_dir, "output_raw.mp4")
                finalize_video(chunk_urls, audio_url, output_path, temp_dir,
                              progress_callback=finalize_progress_callback)

                # Scrub metadata
                update_render_job(supabase_url, supabase_key, render_job_id,
                                  status="muxing", progress=95, message="Removing bot fingerprint metadata...")
                final_path = os.path.join(temp_dir, "output.mp4")
                scrub_cmd = [
                    'ffmpeg', '-y',
                    '-i', output_path,
                    '-map_metadata', '-1',
                    '-bsf:v', 'filter_units=remove_types=6',
                    '-fflags', '+bitexact',
                    '-flags:v', '+bitexact',
                    '-flags:a', '+bitexact',
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    '-movflags', '+faststart',
                    final_path
                ]
                print("Scrubbing metadata...")
                scrub_result = subprocess.run(scrub_cmd, capture_output=True, text=True, timeout=120)
                if scrub_result.returncode != 0:
                    print(f"Metadata scrub warning: {scrub_result.stderr[-200:]}")
                    final_path = output_path
                else:
                    print("Metadata stripped successfully")

                # Upload final video
                update_render_job(supabase_url, supabase_key, render_job_id,
                                  status="uploading", progress=97, message="Uploading final video...")
                storage_path = f"{project_id}/video.mp4"
                video_url = upload_to_supabase(
                    final_path,
                    "generated-assets",
                    storage_path,
                    supabase_url,
                    supabase_key
                )

                elapsed = time.time() - start_time
                print(f"Finalize total time: {elapsed:.1f}s")

                update_render_job(
                    supabase_url, supabase_key, render_job_id,
                    status="complete",
                    progress=100,
                    message=f"Video finalized successfully ({elapsed:.1f}s)",
                    video_url=video_url
                )

                return {
                    "video_url": video_url,
                    "finalize_time_seconds": elapsed
                }

        except Exception as e:
            print(f"Finalize error: {e}")
            update_render_job(
                supabase_url, supabase_key, render_job_id,
                status="failed",
                progress=0,
                message="Finalize failed",
                error=str(e)
            )
            return {"error": str(e)}

    # Regular render mode validation
    if not image_urls:
        return {"error": "No image URLs provided"}
    if not chunk_mode and not audio_url:
        return {"error": "No audio URL provided (required in non-chunk mode)"}
    if not project_id:
        return {"error": "No project ID provided"}
    if not supabase_url or not supabase_key:
        return {"error": "Supabase credentials required"}
    if len(image_urls) != len(timings):
        return {"error": "Image URLs and timings count mismatch"}

    if apply_effects:
        if not os.path.exists(SMOKE_OVERLAY) or not os.path.exists(EMBERS_OVERLAY):
            return {"error": "Overlay files missing - cannot apply effects"}

    cpu_cores = os.cpu_count() or 32
    if chunk_mode:
        print(f"Starting CHUNK render: chunk {chunk_index + 1}/{total_chunks}, {len(image_urls)} images, effects={apply_effects}")
    else:
        print(f"Starting video render: {len(image_urls)} images, effects={apply_effects}, encoder=libx264 (preset={FFMPEG_PRESET}, {cpu_cores} cores)")
    start_time = time.time()

    last_supabase_update = [0]
    def progress_callback(stage: str, percent: float, message: str):
        # In chunk mode, DON'T update the render_job table with progress
        # This prevents race conditions where all 10 workers update the same job
        # The orchestrator tracks chunk completion instead
        if chunk_mode:
            return
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
            # Only update job status in non-chunk mode (orchestrator handles chunk progress)
            if not chunk_mode:
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
                    if completed_count % 10 == 0 and not chunk_mode:
                        dl_pct = 5 + int((completed_count / len(image_urls)) * 15)
                        update_render_job(supabase_url, supabase_key, render_job_id,
                                          status="downloading", progress=dl_pct,
                                          message=f"Downloaded {completed_count}/{len(image_urls)} images")

            if failed_downloads:
                return {"error": f"Failed to download images: {failed_downloads[:5]}"}

            download_elapsed = time.time() - download_start
            print(f"Downloaded {len(image_paths)} images in {download_elapsed:.1f}s")

            # In chunk mode, skip audio download
            audio_path = None
            if not chunk_mode:
                update_render_job(supabase_url, supabase_key, render_job_id,
                                  status="downloading", progress=22, message="Downloading audio...")
                audio_path = os.path.join(temp_dir, "audio.wav")
                if not download_file(audio_url, audio_path, timeout=300):
                    return {"error": "Failed to download audio"}
                update_render_job(supabase_url, supabase_key, render_job_id,
                                  status="rendering", progress=25,
                                  message="Starting video render...")
            else:
                # In chunk mode, just print to logs (orchestrator tracks progress)
                print(f"Chunk {chunk_index + 1}: Starting video render...")
            raw_output_path = os.path.join(temp_dir, "output_raw.mp4")
            render_video_cpu(image_paths, timings, audio_path, raw_output_path, apply_effects,
                            progress_callback=progress_callback, chunk_mode=chunk_mode)

            if chunk_mode:
                # Chunk mode: skip metadata scrub (will do on final concatenated video)
                # Upload directly to chunks subfolder
                output_path = raw_output_path
                # Don't update Supabase status in chunk mode - just log
                print(f"Chunk {chunk_index + 1}: Uploading to storage...")
                storage_path = f"{project_id}/chunks/chunk_{chunk_index:02d}.mp4"
                video_url = upload_to_supabase(
                    output_path,
                    "generated-assets",
                    storage_path,
                    supabase_url,
                    supabase_key
                )

                elapsed = time.time() - start_time
                print(f"Chunk {chunk_index + 1} complete in {elapsed:.1f}s")

                # Don't update render_job status to complete in chunk mode
                # (the orchestrator will track chunk completion separately)
                return {
                    "chunk_url": video_url,
                    "chunk_index": chunk_index,
                    "render_time_seconds": elapsed
                }

            else:
                # Full mode: scrub metadata and upload final video
                # Scrub FFmpeg metadata to prevent YouTube bot flagging
                update_render_job(supabase_url, supabase_key, render_job_id,
                                  status="rendering", progress=86, message="Removing bot fingerprint metadata...")
                output_path = os.path.join(temp_dir, "output.mp4")
                scrub_cmd = [
                    'ffmpeg', '-y',
                    '-i', raw_output_path,
                    '-map_metadata', '-1',
                    '-bsf:v', 'filter_units=remove_types=6',
                    '-fflags', '+bitexact',
                    '-flags:v', '+bitexact',
                    '-flags:a', '+bitexact',
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    '-movflags', '+faststart',
                    output_path
                ]
                print("Scrubbing FFmpeg metadata (instant remux)...")
                scrub_result = subprocess.run(scrub_cmd, capture_output=True, text=True, timeout=120)
                if scrub_result.returncode != 0:
                    print(f"Metadata scrub warning: {scrub_result.stderr[-200:]}")
                    output_path = raw_output_path
                else:
                    print("Metadata stripped successfully")
                    if os.path.exists(raw_output_path):
                        os.remove(raw_output_path)

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
        # In chunk mode, don't update Supabase - orchestrator handles failure tracking
        # Each chunk's error is reported via RunPod output
        if not chunk_mode:
            update_render_job(
                supabase_url, supabase_key, render_job_id,
                status="failed",
                progress=0,
                message="CPU render failed",
                error=str(e)
            )
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
