# Save this file as app.py
import os
import cv2
import zipfile
import tempfile
import gradio as gr
import numpy as np
import shutil
from datetime import datetime
from PIL import Image, ImageEnhance
import subprocess
import torch
import gc

# --- Graceful Dependency Imports ---
# These imports are wrapped in try-except blocks to provide clear error messages
# if a dependency is missing, guiding the user to the installation instructions.

try:
    from face2face import Face2Face
except ImportError:
    print("ERROR: 'face2face-lib' is not installed. Please follow the installation instructions in README.md.")
    exit()

try:
    import ffmpeg
except ImportError:
    print("WARNING: 'ffmpeg-python' is not installed. Video processing will work, but the output will have NO AUDIO.")
    print("Please follow the installation instructions in README.md to include audio in processed videos.")
    ffmpeg = None # Set to None so we can check for its existence later

# --- Setup Directories & Global State ---
OUTPUT_DIR = "output"
TEMP_DIR = "temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print("Initializing Face2Face... This may take a moment.")
f2f = Face2Face()
print("Face2Face initialized successfully.")

# Global flag to control processing loops
stop_processing = False

def update_progress(progress, current, total, stage, substage=""):
    """Update Gradio progress bar with detailed stage information."""
    if progress is None: return
    progress(current / total, f"{stage}: {substage} ({current}/{total})")

def resize_for_preview(img, max_size=400):
    """Resize an image for the Gradio gallery preview without quality loss."""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def enhance_face_quality(img, upscale_factor=1.5, sharpen_strength=1.2):
    """Enhance face quality through upscaling, sharpening, and contrast adjustment."""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        orig_size = pil_img.size
        new_size = (int(orig_size[0] * upscale_factor), int(orig_size[1] * upscale_factor))
        
        # High-quality resize and sharpen
        upscaled = pil_img.resize(new_size, Image.LANCZOS)
        enhancer = ImageEnhance.Sharpness(upscaled)
        sharpened = enhancer.enhance(sharpen_strength)
        
        # Subtle contrast enhancement
        contrast_enhancer = ImageEnhance.Contrast(sharpened)
        enhanced = contrast_enhancer.enhance(1.1)
        
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    except Exception:
        # If any part of the enhancement fails, return the original image
        return img

def preprocess_for_quality(img, target_size=1024):
    """Preprocess image to a minimum size for higher quality face swapping."""
    h, w = img.shape[:2]
    if max(h, w) < target_size:
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img

def cleanup_temp():
    """Clean up the temporary directory on startup."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except OSError as e:
            print(f"Error cleaning up temp directory: {e}")
    os.makedirs(TEMP_DIR, exist_ok=True)

def open_output_dir():
    """Open the output directory in the system's file explorer."""
    abs_path = os.path.abspath(OUTPUT_DIR)
    try:
        if os.name == 'nt':  # Windows
            os.startfile(abs_path)
        elif os.uname().sysname == 'Darwin':  # macOS
            subprocess.run(['open', abs_path])
        else:  # Linux
            subprocess.run(['xdg-open', abs_path])
    except Exception as e:
        gr.Warning(f"Could not open output directory automatically. Please navigate to: {abs_path}. Error: {e}")

def process_files(src_img, tgt_files, enhance, quality_mode, upscale_factor, progress=gr.Progress()):
    """Process a batch of images."""
    global stop_processing
    stop_processing = False

    if src_img is None or not tgt_files:
        gr.Warning("Please upload a source face and at least one target image.")
        return [], None

    # Create a unique session directory for this batch
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_temp = os.path.join(TEMP_DIR, f"session_img_{session_id}")
    os.makedirs(session_temp, exist_ok=True)
    
    zip_path = os.path.join(OUTPUT_DIR, f"results_images_{session_id}.zip")
    output_images_for_gallery = []

    try:
        # Pre-process the source image once
        src_temp_path = os.path.join(session_temp, "source.jpg")
        src_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
        src_to_save = preprocess_for_quality(src_bgr) if quality_mode else src_bgr
        cv2.imwrite(src_temp_path, src_to_save)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            total_images = len(tgt_files)
            for idx, tgt_file in enumerate(tgt_files):
                if stop_processing:
                    gr.Info("Processing stopped by user.")
                    break

                current_image_num = idx + 1
                update_progress(progress, current_image_num, total_images, "Processing Images", f"Image {current_image_num}/{total_images}")

                try:
                    # Read and process target image
                    tgt_img_bgr = cv2.imread(tgt_file.name)
                    if tgt_img_bgr is None: continue

                    tgt_temp_path = os.path.join(session_temp, f"target_{idx}.jpg")
                    tgt_to_save = preprocess_for_quality(tgt_img_bgr) if quality_mode else tgt_img_bgr
                    cv2.imwrite(tgt_temp_path, tgt_to_save)
                    
                    # Perform the swap
                    result = f2f.swap_img_to_img(src_temp_path, tgt_temp_path)
                    
                    # Apply enhancements if selected
                    if quality_mode:
                        result = enhance_face_quality(result, upscale_factor)
                    if enhance:
                        result = f2f.enhance_faces(result)
                    
                    # Save result and add to zip
                    result_filename = f"result_{session_id}_{idx+1:03d}.jpg"
                    result_path = os.path.join(OUTPUT_DIR, result_filename)
                    cv2.imwrite(result_path, result)
                    zipf.write(result_path, arcname=result_filename)
                    
                    # Prepare image for Gradio gallery preview
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    output_images_for_gallery.append(resize_for_preview(result_rgb))

                except Exception as e:
                    print(f"Failed to process image {tgt_file.name}: {e}")
                    continue # Skip to the next image
                finally:
                    # Clean up GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        update_progress(progress, 1, 1, "Complete!", "Batch processing finished.")
        return output_images_for_gallery, zip_path

    except Exception as e:
        gr.Error(f"An unexpected error occurred: {e}")
        return [], None
    finally:
        shutil.rmtree(session_temp, ignore_errors=True)


def process_video(src_img, video_path, enhance, quality_mode, upscale_factor, progress=gr.Progress(), parent_progress_info=""):
    """Process a single video file frame by frame."""
    global stop_processing
    
    if src_img is None or not video_path:
        gr.Warning("Please upload a source face and a target video.")
        return None
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_temp = os.path.join(TEMP_DIR, f"session_vid_{session_id}")
    os.makedirs(session_temp, exist_ok=True)
    
    # Paths for temporary and final files
    src_temp_path = os.path.join(session_temp, "src.jpg")
    temp_video_path = os.path.join(session_temp, "temp_video.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"result_video_{session_id}.mp4")

    # Prepare source image
    src_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    src_to_save = preprocess_for_quality(src_bgr) if quality_mode else src_bgr
    cv2.imwrite(src_temp_path, src_to_save)

    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            gr.Error("Could not open video file.")
            return None
            
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            gr.Error("Could not create video writer. Your OpenCV install may lack codecs.")
            return None

        # Frame processing loop
        frame_count = 0
        while cap.isOpened():
            if stop_processing:
                gr.Info("Processing stopped by user.")
                break
                
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            update_progress(progress, frame_count, total_frames, f"{parent_progress_info}Processing Video", f"Frame {frame_count}/{total_frames}")

            tgt_temp_path = os.path.join(session_temp, "tgt_frame.jpg")
            tgt_to_save = preprocess_for_quality(frame) if quality_mode else frame
            cv2.imwrite(tgt_temp_path, tgt_to_save)
            
            result = f2f.swap_img_to_img(src_temp_path, tgt_temp_path)
            
            if quality_mode: result = enhance_face_quality(result, upscale_factor)
            if enhance: result = f2f.enhance_faces(result)
            
            # Resize result back to original video dimensions and write to file
            final_frame = cv2.resize(result, (width, height))
            out.write(final_frame)
            
            if frame_count % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    except Exception as e:
        gr.Error(f"Error during video frame processing: {e}")
        return None
    finally:
        if cap: cap.release()
        if out: out.release()

    if stop_processing:
        shutil.rmtree(session_temp, ignore_errors=True)
        return None

    # --- Finalizing Video (Adding Audio) ---
    update_progress(progress, 1, 1, f"{parent_progress_info}Finalizing", "Adding audio...")
    if ffmpeg and os.path.exists(temp_video_path):
        try:
            # Check if original video has an audio stream
            probe = ffmpeg.probe(video_path)
            if any(s['codec_type'] == 'audio' for s in probe['streams']):
                input_video = ffmpeg.input(temp_video_path)
                input_audio = ffmpeg.input(video_path)
                ffmpeg.output(input_video.video, input_audio.audio, output_path, vcodec='copy', acodec='aac', shortest=None).run(overwrite_output=True, quiet=True)
            else:
                # No audio stream, just copy the file
                gr.Info("Original video has no audio. Output will be silent.")
                shutil.copy2(temp_video_path, output_path)
        except Exception as e:
            gr.Warning(f"Could not add audio due to FFmpeg error: {e}. Output video will be silent. Make sure FFmpeg is installed correctly.")
            shutil.copy2(temp_video_path, output_path)
    elif os.path.exists(temp_video_path):
        gr.Warning("`ffmpeg-python` not found or not working. Output video will be silent.")
        shutil.copy2(temp_video_path, output_path)
    
    shutil.rmtree(session_temp, ignore_errors=True)
    update_progress(progress, 1, 1, f"{parent_progress_info}Complete!", "Video processing finished.")
    return output_path


def process_batch_videos(src_img, video_files, enhance, quality_mode, upscale_factor, progress=gr.Progress()):
    """Process multiple video files and zip the results."""
    global stop_processing
    stop_processing = False

    if src_img is None or not video_files:
        gr.Warning("Please upload a source face and at least one target video.")
        return None

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(OUTPUT_DIR, f"results_videos_{session_id}.zip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        total_videos = len(video_files)
        for idx, video_file in enumerate(video_files):
            if stop_processing:
                gr.Info("Processing stopped by user.")
                break
            
            current_video_num = idx + 1
            progress_prefix = f"Video {current_video_num}/{total_videos}: "
            update_progress(progress, current_video_num, total_videos, "Batch Progress", os.path.basename(video_file.name))
            
            result_path = process_video(src_img, video_file.name, enhance, quality_mode, upscale_factor, progress, parent_progress_info=progress_prefix)
            
            if result_path and os.path.exists(result_path):
                zipf.write(result_path, arcname=os.path.basename(result_path))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
    update_progress(progress, 1, 1, "Complete!", "Batch video processing finished.")
    return zip_path


def stop_processing_fn():
    """Sets the global flag to stop all processing loops."""
    global stop_processing
    stop_processing = True
    gr.Info("Stop signal received. Processing will halt after the current item.")


# --- Gradio UI Definition ---
with gr.Blocks(title="NexFace Swapper", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ NexFace: High-Quality Face Swapper")
    gr.Markdown("Swap faces in images and videos with advanced quality enhancement options.")

    with gr.Tabs():
        # --- Image Processing Tab ---
        with gr.TabItem("🖼️ Image-to-Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Source Face")
                    img_src = gr.Image(type="numpy", label="Source Face", image_mode="RGB")
                with gr.Column(scale=2):
                    gr.Markdown("### 🧑‍🤝‍🧑 Target Images")
                    img_tgt = gr.File(file_count="multiple", file_types=["image"], label="Target Images")
            
            with gr.Accordion("⚙️ Processing Options", open=False):
                img_enhance = gr.Checkbox(label="✨ Enhance Output Faces", value=True, info="Applies a final enhancement pass to improve face details.")
                img_quality = gr.Checkbox(label="🔍 High Quality Mode", value=True, info="Pre-upscales images for better detection and applies post-swap sharpening.")
                img_upscale_factor = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Face Upscale Factor", info="Controls the upscaling strength in High Quality Mode.")
            
            with gr.Row():
                img_start_btn = gr.Button("Start Processing", variant="primary", scale=3)
                img_stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)

            gr.Markdown("### 📸 Results")
            with gr.Row():
                img_gallery = gr.Gallery(columns=5, height="auto", object_fit="contain", label="Swapped Results")
                img_zip_output = gr.File(label="Download All as .zip")
            
            img_open_dir_btn = gr.Button("📂 Open Output Directory", variant="secondary")

        # --- Video Processing Tab ---
        with gr.TabItem("🎥 Video-to-Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Source Face")
                    vid_src = gr.Image(type="numpy", label="Source Face", image_mode="RGB")
                with gr.Column(scale=2):
                    gr.Markdown("### 🎬 Target Video")
                    vid_tgt = gr.Video(label="Target Video")
            
            with gr.Accordion("⚙️ Processing Options", open=False):
                vid_enhance = gr.Checkbox(label="✨ Enhance Output Faces", value=True, info="Applies a final enhancement pass to improve face details.")
                vid_quality = gr.Checkbox(label="🔍 High Quality Mode", value=True, info="Pre-upscales frames for better detection and applies post-swap sharpening.")
                vid_upscale_factor = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Face Upscale Factor", info="Controls the upscaling strength in High Quality Mode.")

            with gr.Row():
                vid_start_btn = gr.Button("Start Processing", variant="primary", scale=3)
                vid_stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)
            
            gr.Markdown("### 🎬 Result")
            vid_output = gr.Video(label="Processed Video")
            vid_open_dir_btn = gr.Button("📂 Open Output Directory", variant="secondary")

        # --- Batch Video Processing Tab ---
        with gr.TabItem("🎬 Batch Video Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Source Face")
                    batch_vid_src = gr.Image(type="numpy", label="Source Face", image_mode="RGB")
                with gr.Column(scale=2):
                    gr.Markdown("### 🎬 Target Videos")
                    batch_vid_tgt = gr.File(file_count="multiple", file_types=["video"], label="Target Videos")

            with gr.Accordion("⚙️ Processing Options", open=False):
                batch_vid_enhance = gr.Checkbox(label="✨ Enhance Output Faces", value=True, info="Applies a final enhancement pass to improve face details.")
                batch_vid_quality = gr.Checkbox(label="🔍 High Quality Mode", value=True, info="Pre-upscales frames for better detection and applies post-swap sharpening.")
                batch_vid_upscale_factor = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Face Upscale Factor", info="Controls the upscaling strength in High Quality Mode.")

            with gr.Row():
                batch_vid_start_btn = gr.Button("Start Batch Processing", variant="primary", scale=3)
                batch_vid_stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)

            gr.Markdown("### 📦 Result")
            batch_zip_output = gr.File(label="Download All as .zip")
            batch_open_dir_btn = gr.Button("📂 Open Output Directory", variant="secondary")

    # --- Event Handlers ---
    # Image Tab
    img_start_btn.click(fn=process_files, inputs=[img_src, img_tgt, img_enhance, img_quality, img_upscale_factor], outputs=[img_gallery, img_zip_output])
    img_stop_btn.click(fn=stop_processing_fn, queue=False)
    img_open_dir_btn.click(fn=open_output_dir, queue=False)

    # Video Tab
    vid_start_btn.click(fn=process_video, inputs=[vid_src, vid_tgt, vid_enhance, vid_quality, vid_upscale_factor], outputs=[vid_output])
    vid_stop_btn.click(fn=stop_processing_fn, queue=False)
    vid_open_dir_btn.click(fn=open_output_dir, queue=False)

    # Batch Video Tab
    batch_vid_start_btn.click(fn=process_batch_videos, inputs=[batch_vid_src, batch_vid_tgt, batch_vid_enhance, batch_vid_quality, batch_vid_upscale_factor], outputs=[batch_zip_output])
    batch_vid_stop_btn.click(fn=stop_processing_fn, queue=False)
    batch_open_dir_btn.click(fn=open_output_dir, queue=False)


if __name__ == '__main__':
    cleanup_temp()
    demo.queue().launch()