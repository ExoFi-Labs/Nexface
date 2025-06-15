# 👁️ NexFace: High-Quality Face Swapper

A user-friendly desktop application for swapping faces in images and videos, built with Python and Gradio. This tool supports single or batch processing for both images and videos, with built-in options for enhancing the quality of the final output.

  <!-- You should replace this with an actual screenshot of your app -->

## Features

-   **Image-to-Image Swapping**: Swap a source face onto one or multiple target images.
-   **Video-to-Video Swapping**: Swap a source face into every frame of a target video.
-   **Batch Processing**: Process multiple target images or videos in a single run.
-   **Quality Enhancement**: Optional modes to improve the realism and detail of swapped faces.
-   **Audio Preservation**: Automatically retains the original audio track in processed videos.
-   **User-Friendly Interface**: Simple, tab-based UI for easy operation.

## ⚠️ Prerequisites

Before you begin, you **must** have the following software installed on your system.

1.  **Python**: Version 3.9 or newer. You can download it from [python.org](https://www.python.org/).
2.  **Git**: Required to clone this repository. You can get it from [git-scm.com](https://git-scm.com/).
3.  **FFmpeg**: **(CRITICAL FOR VIDEO)** This is required to process videos and add audio to the output.
    -   **Windows**: Download a build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (the `essentials` build is fine). Unzip it, and you **must** add the `bin` folder to your system's PATH environment variable.
    -   **macOS**: Use Homebrew: `brew install ffmpeg`
    -   **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install ffmpeg`
4.  **(Optional but Recommended) NVIDIA GPU with CUDA**: For significant processing speed-up, an NVIDIA GPU is highly recommended.
    -   Ensure you have the latest [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) installed.
    -   Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive). The installation guide below assumes **CUDA 12.1**, but you should use the command that matches your installed CUDA version.

## ⚙️ Installation Instructions

Follow these steps carefully in your terminal or command prompt. **Do not skip steps.**

### Step 1: Clone the Repository

Clone this project to your local machine.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2: Create and Activate a Virtual Environment

This is a crucial step to avoid conflicts with other Python projects.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

You will know it's active when you see `(venv)` at the beginning of your command prompt line.

### Step 3: Install Core ML Libraries (PyTorch & ONNXRuntime)

This is the most common point of failure. **Install these manually** based on your hardware.

**A) For NVIDIA GPU Users (Highly Recommended):**

Run the following command. This example is for CUDA 12.1. Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the correct command for your specific CUDA version if it's different.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu
```

**B) For CPU-Only Users:**

If you do not have an NVIDIA GPU, the application will be much slower, but it will work.

```bash
pip install torch torchvision torchaudio
pip install onnxruntime
```

### Step 4: Install Remaining Python Packages

Now that the core libraries are installed, you can install the rest from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

The installation is now complete!

## ▶️ Running the Application

With your virtual environment still active, run the following command:

```bash
python app.py
```

This will start the Gradio web server. A local URL (like `http://127.0.0.1:7860`) will be displayed in your terminal. Open this URL in your web browser to use the application.

## 💡 Usage Guide

1.  **Select a Tab**: Choose between `Image-to-Image`, `Video-to-Video`, or `Batch Video Processing`.
2.  **Upload Source Face**: In the left panel, upload a clear, front-facing image of the face you want to use.
3.  **Upload Target(s)**: In the right panel, upload the image(s) or video(s) you want to swap the face onto.
4.  **Set Options**:
    -   `Enhance Output Faces`: Recommended for a higher quality result.
    -   `High Quality Mode`: Slower, but pre-processes inputs for better results.
5.  **Start Processing**: Click the "Start Processing" button.
6.  **View & Download**: Results will appear in the gallery or video player. A `.zip` file of all results (for batch jobs) or the final video can be downloaded. You can also click "Open Output Directory" to directly access all saved files.

## Troubleshooting

-   **`ModuleNotFoundError: No module named 'face2face'` or `'torch'`**: You either skipped a step or the installation failed. Deactivate and delete the `venv` folder, then start the installation guide from Step 2.
-   **`ffmpeg: command not found` or Video Output is Silent**: This means FFmpeg is not installed or not in your system's PATH. Go back to the **Prerequisites** section and ensure FFmpeg is installed correctly.
-   **CUDA Errors (e.g., `CUDA out of memory`)**: Your GPU may not have enough VRAM. Try disabling "High Quality Mode", reducing the "Face Upscale Factor", or processing fewer files at once.
-   **Application is Very Slow**: You are likely using the CPU version. GPU acceleration is essential for fast processing. Ensure you followed the GPU installation steps and that your system recognizes your NVIDIA GPU.