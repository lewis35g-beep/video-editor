import os
import json
import base64
import requests
from pathlib import Path

import streamlit as st
from openai import OpenAI
from moviepy import ImageClip, VideoFileClip, AudioFileClip, concatenate_videoclips

# =====================================================
# AI VIDEO EDITOR WITH PEXELS B-ROLL
# =====================================================
# Required installs:
# pip install streamlit openai moviepy requests
#
# Required environment variables:
# OPENAI_API_KEY=your_openai_key
# PEXELS_API_KEY=your_pexels_key
#
# Run:
# streamlit run ai_video_editor_with_pexels.py
# =====================================================

APP_DIR = Path("ai_video_editor_output")
SCENE_DIR = APP_DIR / "scenes"
AUDIO_DIR = APP_DIR / "audio"
EXPORT_DIR = APP_DIR / "exports"
BROLL_DIR = APP_DIR / "broll"
UPLOAD_DIR = APP_DIR / "uploads"

for folder in [APP_DIR, SCENE_DIR, AUDIO_DIR, EXPORT_DIR, BROLL_DIR, UPLOAD_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

st.set_page_config(page_title="AI Video Creator", layout="wide")
st.title("AI Video Creator / Editor with Pexels B-roll")
st.write("Paste a script, let AI build scenes, pull real B-roll from Pexels, generate voiceover, and export an MP4.")

# =====================================================
# HELPERS
# =====================================================

def clean_json(raw: str):
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def safe_filename(text: str, max_length: int = 60):
    cleaned = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in text.lower())
    return cleaned[:max_length].strip("_") or "clip"

# =====================================================
# OPENAI: SCRIPT TO SCENES
# =====================================================

def split_script_into_scenes(script: str, target_scene_count: int = 6):
    prompt = f"""
You are an AI video editor.
Split the script into approximately {target_scene_count} cinematic scenes.
Return ONLY valid JSON.

Each scene must include:
- scene_number
- narration
- visual_prompt
- broll_search
- duration_seconds
- mood
- on_screen_text

Rules:
- broll_search must be a short stock-video search phrase.
- Example broll_search: "empty city street night", "police lights dark road", "courtroom trial", "storm clouds over highway".
- visual_prompt can be more detailed for AI image fallback.
- Keep scene durations between 4 and 8 seconds.

Script:
{script}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return clean_json(response.output_text)

# =====================================================
# PEXELS B-ROLL SEARCH
# =====================================================

def search_pexels_videos(query: str, per_page: int = 5):
    if not PEXELS_API_KEY:
        raise ValueError("Missing PEXELS_API_KEY environment variable.")

    url = "https://api.pexels.com/videos/search"
    params = {
        "query": query,
        "orientation": "landscape",
        "min_width": 1280,
        "min_height": 720,
        "min_duration": 4,
        "max_duration": 30,
        "per_page": per_page,
        "page": 1,
    }

    response = requests.get(
        url,
        headers={"Authorization": PEXELS_API_KEY},
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("videos", [])


def choose_best_pexels_file(video: dict):
    files = video.get("video_files", [])
    mp4_files = [f for f in files if f.get("file_type") == "video/mp4"]

    if not mp4_files:
        return None

    mp4_files = sorted(
        mp4_files,
        key=lambda f: ((f.get("width") or 0), (f.get("height") or 0)),
        reverse=True,
    )

    for file in mp4_files:
        if (file.get("width") or 0) >= 1280 and (file.get("height") or 0) >= 720:
            return file

    return mp4_files[0]


def download_pexels_video(video: dict, scene_number: int, query: str):
    selected_file = choose_best_pexels_file(video)
    if not selected_file:
        return None

    video_url = selected_file["link"]
    filename = f"scene_{scene_number:02d}_{safe_filename(query)}.mp4"
    output_path = BROLL_DIR / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    with requests.get(video_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    metadata = {
        "source": "pexels",
        "scene_number": scene_number,
        "search_query": query,
        "pexels_id": video.get("id"),
        "pexels_url": video.get("url"),
        "duration": video.get("duration"),
        "creator": video.get("user", {}),
        "license_note": "Check Pexels license/API terms for your specific use case.",
    }

    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def get_broll_for_scene(scene: dict):
    query = scene.get("broll_search") or scene.get("visual_prompt") or scene.get("narration")
    videos = search_pexels_videos(query, per_page=5)

    if not videos:
        return None

    best_video = videos[0]
    return download_pexels_video(best_video, scene["scene_number"], query)

# =====================================================
# OPENAI IMAGE FALLBACK
# =====================================================

def generate_scene_image(prompt: str, scene_number: int):
    image_prompt = f"""
Create a cinematic 16:9 YouTube video scene.
Style: realistic documentary, dramatic lighting, high detail, professional composition.
Scene: {prompt}
No text, no logos, no watermarks.
"""

    result = client.images.generate(
        model="gpt-image-1",
        prompt=image_prompt,
        size="1536x864",
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    output_path = SCENE_DIR / f"scene_{scene_number:02d}.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    return output_path

# =====================================================
# OPENAI VOICEOVER
# =====================================================

def generate_voiceover(text: str, filename: str = "voiceover.mp3"):
    output_path = AUDIO_DIR / filename

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=text,
    )

    speech.write_to_file(output_path)
    return output_path

# =====================================================
# VIDEO ASSEMBLY
# =====================================================

def make_video_clip_from_media(media_path: Path, duration: float):
    suffix = media_path.suffix.lower()

    if suffix in [".mp4", ".mov", ".m4v", ".webm"]:
        clip = VideoFileClip(str(media_path))

        if clip.duration > duration:
            clip = clip.subclipped(0, duration)
        else:
            clip = clip.with_duration(duration)

        clip = clip.without_audio()
    else:
        clip = ImageClip(str(media_path)).with_duration(duration)

    clip = clip.resized(height=1080)

    if clip.w < 1920:
        clip = clip.resized(width=1920)

    clip = clip.cropped(
        x_center=clip.w / 2,
        y_center=clip.h / 2,
        width=1920,
        height=1080,
    )

    return clip


def make_video_from_media(scenes, media_paths, audio_path=None, output_name="final_video.mp4"):
    clips = []

    for scene, media_path in zip(scenes, media_paths):
        duration = float(scene.get("duration_seconds", 5))
        clip = make_video_clip_from_media(Path(media_path), duration)
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")

    if audio_path:
        audio = AudioFileClip(str(audio_path))
        final = final.with_audio(audio)

    output_path = EXPORT_DIR / output_name
    final.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
    )

    return output_path

# =====================================================
# STREAMLIT UI
# =====================================================

with st.sidebar:
    st.header("Settings")
    scene_count = st.slider("Approximate number of scenes", 3, 12, 6)
    use_pexels = st.checkbox("Use real Pexels B-roll", value=True)
    generate_images = st.checkbox("Use AI image fallback", value=True)
    generate_audio = st.checkbox("Generate AI voiceover", value=True)

    st.divider()
    st.write("Environment variables needed:")
    st.code("OPENAI_API_KEY=your_openai_key\nPEXELS_API_KEY=your_pexels_key")

script = st.text_area(
    "Paste your video script",
    height=300,
    placeholder="Paste your YouTube narration script here...",
)

uploaded_files = st.file_uploader(
    "Optional: upload your own images for future smart placement",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        path = UPLOAD_DIR / file.name
        with open(path, "wb") as f:
            f.write(file.read())
    st.info(f"Uploaded {len(uploaded_files)} image(s). Smart placement can be added next.")

if "scenes" not in st.session_state:
    st.session_state.scenes = None

if "media_paths" not in st.session_state:
    st.session_state.media_paths = []

if st.button("1. Create AI Video Plan"):
    if not script.strip():
        st.error("Paste a script first.")
    else:
        with st.spinner("AI is splitting your script into scenes and B-roll searches..."):
            st.session_state.scenes = split_script_into_scenes(script, scene_count)
        st.success("Scene plan created.")

if st.session_state.scenes:
    st.subheader("AI Scene Plan")

    for scene in st.session_state.scenes:
        with st.expander(f"Scene {scene['scene_number']}: {scene.get('mood', '')}", expanded=True):
            st.write("Narration:")
            st.write(scene["narration"])
            st.write("B-roll Search:")
            st.code(scene.get("broll_search", ""))
            st.write("AI Image Prompt:")
            st.write(scene["visual_prompt"])
            st.write("On-screen text:")
            st.write(scene.get("on_screen_text", ""))
            st.write(f"Duration: {scene.get('duration_seconds', 5)} seconds")

    if st.button("2. Get B-roll / Scene Media"):
        media_paths = []
        progress = st.progress(0)

        for i, scene in enumerate(st.session_state.scenes):
            scene_number = scene["scene_number"]
            media_path = None

            if use_pexels:
                try:
                    with st.spinner(f"Searching Pexels B-roll for scene {scene_number}..."):
                        media_path = get_broll_for_scene(scene)
                except Exception as e:
                    st.warning(f"Pexels failed for scene {scene_number}: {e}")

            if not media_path and generate_images:
                with st.spinner(f"Generating AI fallback image for scene {scene_number}..."):
                    media_path = generate_scene_image(scene["visual_prompt"], scene_number)

            if not media_path:
                st.error(f"No media found for scene {scene_number}.")
                st.stop()

            media_paths.append(media_path)
            progress.progress((i + 1) / len(st.session_state.scenes))

        st.session_state.media_paths = media_paths
        st.success("Scene media ready.")

if st.session_state.media_paths:
    st.subheader("Selected Scene Media")

    for path in st.session_state.media_paths:
        path = Path(path)
        st.write(path.name)
        if path.suffix.lower() in [".mp4", ".mov", ".m4v", ".webm"]:
            st.video(str(path))
        else:
            st.image(str(path), caption=path.name)

    if st.button("3. Generate Voiceover and Export Video"):
        full_narration = "\n\n".join(scene["narration"] for scene in st.session_state.scenes)
        audio_path = None

        if generate_audio:
            with st.spinner("Generating voiceover..."):
                audio_path = generate_voiceover(full_narration)

        with st.spinner("Rendering final MP4..."):
            output_path = make_video_from_media(
                st.session_state.scenes,
                st.session_state.media_paths,
                audio_path=audio_path,
            )

        st.success("Video exported.")
        st.video(str(output_path))

        with open(output_path, "rb") as f:
            st.download_button(
                "Download final video",
                data=f,
                file_name="final_video.mp4",
                mime="video/mp4",
            )
