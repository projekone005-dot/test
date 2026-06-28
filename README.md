from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import gc
import traceback
import urllib.error
import torch
from diffusers import StableDiffusionPipeline
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from llama_cpp import Llama
from tkinter import BOTH, DISABLED, END, LEFT, NORMAL, RIGHT, TOP, X, Y, filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except Exception:  # pragma: no cover - shown as a UI error in main()
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageTk = None


APP_NAME = "AI ArAnimator"

BASE_DIR = Path(__file__).resolve().parent
APP_HOME = Path(os.environ.get("ARANIMATOR_HOME", str(BASE_DIR))).resolve()

APP_DATA_DIR = APP_HOME / "data"
OUTPUT_DIR = APP_DATA_DIR / "outputs"

MODELS_DIR = APP_HOME / "models" / "llm"
DIFFUSION_MODELS_DIR = APP_HOME / "models" / "diffusion"
VIDEO_MODELS_DIR = APP_HOME / "models" / "video"

LOCAL_TOOLS_DIR = APP_HOME / "tools"

DEFAULT_LLM = "qwen3-4b-q4_k_m.gguf"

LLM_CONTEXT = 4096

# Sisakan 2 thread untuk Windows
LLM_THREADS = max(1, (os.cpu_count() or 4) - 2)

LLM_BATCH = 256

# CPU Only
LLM_GPU_LAYERS = 0


@dataclass
class Scene:
    number: int
    title: str
    duration: float = 3.0

    setting: dict = field(default_factory=lambda: {
        "location":"","time":"","weather":"","season":"",
        "lighting":"","environment":[],"environment_motion":[]
    })

    scene_action: list[str] = field(default_factory=list)

    characters: list[dict] = field(default_factory=list)

    camera: dict = field(default_factory=lambda: {
        "shot":"","angle":"","movement":"","focus":""
    })

    mood: dict = field(default_factory=lambda: {
        "emotion":"","atmosphere":"",
        "lighting":"","color_tone":""
    })

    narration: str = ""
    director_notes: list[str] = field(default_factory=list)


@dataclass
class Storyboard:
    title: str
    logline: str
    style: str
    scenes: list[Scene]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "logline": self.logline,
            "style": self.style,
            "scenes": [scene.__dict__ for scene in self.scenes],
        }


def slugify(value: str, fallback: str = "animasi") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:48] or fallback


def safe_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return min(8.0, max(2.0, result))

def model_path(model_name: str) -> Path:
    return MODELS_DIR / model_name


def list_local_models() -> list[str]:
    """
    Mengembalikan semua model GGUF di folder models.
    """

    if not MODELS_DIR.exists():
        return []

    return sorted(
        [
            file.name
            for file in MODELS_DIR.glob("*.gguf")
            if file.is_file()
        ]
    )


class LocalLLM:
    def __init__(self, model: str = DEFAULT_LLM):
        self.model_name = model or DEFAULT_LLM
        self.model_path = model_path(self.model_name)

        self.llm = None
        self.loaded = False

        self.load_model()

    def load_model(self) -> None:
        """
        Memuat model GGUF ke RAM.
        Hanya dipanggil sekali.
        """

        if self.loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan:\n{self.model_path}"
            )

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=LLM_CONTEXT,
            n_threads=LLM_THREADS,
            n_batch=LLM_BATCH,
            n_gpu_layers=LLM_GPU_LAYERS,
            verbose=False,
        )

        self.loaded = True

    def unload(self):
        """
        Membebaskan RAM jika model ingin diganti.
        """

        if self.llm is not None:
            del self.llm

        self.llm = None
        self.loaded = False

        gc.collect()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2200,
    ) -> str:
        """
        Mengirim prompt ke Qwen lokal.
        """

        if not self.loaded:
            self.load_model()

        response = self.llm.create_chat_completion(

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a film director.\n"
                        "Output JSON only.\n"
                        "Never explain.\n"
                        "Never think.\n"
                        "Never output <think>.\n"
                        "Start immediately with {"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0.2,
            max_tokens=2200,

        )

        return (
            response["choices"][0]
            ["message"]
            ["content"]
            .strip()
        )

    def create_storyboard(
        self,
        user_prompt: str,
    ) -> Storyboard:

        prompt = self._build_prompt(user_prompt)

        try:

            raw = self.generate(
                prompt,
                temperature=0.2,
                max_tokens=900,
            )

            print("=" * 80)
            print(raw)
            print("=" * 80)

            parsed = self._parse_json(raw)

            storyboard = normalize_storyboard(
                parsed,
                user_prompt,
            )

            self.last_story_prompt = user_prompt

            return storyboard

        except Exception:

            print("=" * 60)
            print("LOCAL LLM ERROR")
            print(traceback.format_exc())
            print("=" * 60)

            return fallback_storyboard(user_prompt)

    def expand_scene(
        self,
        storyboard: Storyboard,
        scene: Scene,
    ) -> Scene:

        prompt = f"""
    Anda adalah Sutradara Film Animasi profesional.

    Balas HANYA JSON.

    Jangan markdown.
    Jangan komentar.
    Jangan penjelasan.

    Lengkapi SATU scene berikut.

    Cerita:
    {storyboard.logline}

    Style:
    {storyboard.style}

    Scene:
    {scene.title}

    Setting awal:
    {json.dumps(scene.setting, ensure_ascii=False)}

    Scene Action:
    {json.dumps(scene.scene_action, ensure_ascii=False)}

    Karakter:
    {json.dumps(scene.characters, ensure_ascii=False)}

    Lengkapi menjadi JSON berikut.

    {{
    "setting":{{
    "location":"",
    "time":"",
    "weather":"",
    "season":"",
    "lighting":"",
    "environment":[],
    "environment_motion":[]
    }},

    "characters":[
    {{
    "name":"",
    "role":"",
    "gender":"",
    "age":"",
    "identity":"",
    "appearance":"",
    "clothes":"",
    "expression":"",
    "action":"",
    "natural_motion":[],
    "uniqueness":""
    }}
    ],

    "camera":{{
    "shot":"",
    "angle":"",
    "movement":"",
    "focus":""
    }},

    "mood":{{
    "emotion":"",
    "atmosphere":"",
    "lighting":"",
    "color_tone":""
    }},

    "scene_action":[],

    "director_notes":[]
    }}

    Aturan:

    - Pertahankan karakter yang sudah ada.
    - Jangan mengganti nama karakter.
    - Jangan mengubah identitas karakter.
    - Jika ada lebih dari satu karakter maka semua harus tampil.
    - Lengkapi appearance bila masih kurang.
    - Tambahkan natural_motion yang alami.
    - Tambahkan environment yang kaya.
    - Tambahkan environment_motion sehingga dunia terasa hidup.
    - Camera harus sinematik.
    - Director_notes maksimal 4 poin.
    - Balas JSON saja.
    """.strip()

        try:

            raw = self.generate(
                prompt,
                temperature=0.35,
                max_tokens=900,
            )

            print("=" * 80)
            print(raw)
            print("=" * 80)

            data = self._parse_json(raw)

            if "setting" in data:
                scene.setting = data["setting"]

            if "characters" in data:
                scene.characters = data["characters"]

            if "camera" in data:
                scene.camera = data["camera"]

            if "mood" in data:
                scene.mood = data["mood"]

            if "scene_action" in data:
                scene.scene_action = data["scene_action"]

            if "director_notes" in data:
                scene.director_notes = data["director_notes"]

            return scene

        except Exception:

            print("=" * 60)
            print("EXPAND SCENE ERROR")
            print(traceback.format_exc())
            print("=" * 60)

            return scene

    def normalize_scene_details(data: dict, scene: Scene) -> Scene:
        """
        Menggabungkan hasil expand_scene() ke scene lama.
        Tidak mengubah identitas karakter.
        """

        if not isinstance(data, dict):
            return scene

        # ========= Setting =========

        setting = data.get("setting")
        if isinstance(setting, dict):
            scene.setting.update({
                "weather": setting.get("weather", scene.setting.get("weather")),
                "season": setting.get("season", scene.setting.get("season")),
                "lighting": setting.get("lighting", scene.setting.get("lighting")),
                "environment": setting.get("environment", scene.setting.get("environment", [])),
                "environment_motion": setting.get("environment_motion", scene.setting.get("environment_motion", [])),
            })

        # ========= Scene Action =========

        actions = data.get("scene_action")
        if isinstance(actions, list) and actions:
            scene.scene_action = [str(x) for x in actions]

        # ========= Camera =========

        camera = data.get("camera")
        if isinstance(camera, dict):
            scene.camera.update(camera)

        # ========= Mood =========

        mood = data.get("mood")
        if isinstance(mood, dict):
            scene.mood.update(mood)

        # ========= Director Notes =========

        notes = data.get("director_notes")
        if isinstance(notes, list) and notes:
            scene.director_notes = [str(x) for x in notes]

        # ========= Narration =========

        narration = data.get("narration")
        if narration:
            scene.narration = str(narration)

        # ========= Character Detail =========

        new_chars = data.get("characters")

        if isinstance(new_chars, list):

            for old_char, new_char in zip(scene.characters, new_chars):

                if not isinstance(new_char, dict):
                    continue

                # identitas tetap
                old_char["name"] = new_char.get("name", old_char.get("name"))
                old_char["role"] = new_char.get("role", old_char.get("role"))
                old_char["gender"] = new_char.get("gender", old_char.get("gender"))
                old_char["age"] = new_char.get("age", old_char.get("age"))
                old_char["identity"] = new_char.get("identity", old_char.get("identity"))

                # boleh diperkaya
                old_char["appearance"] = new_char.get("appearance", old_char.get("appearance"))
                old_char["clothes"] = new_char.get("clothes", old_char.get("clothes"))
                old_char["expression"] = new_char.get("expression", old_char.get("expression"))
                old_char["action"] = new_char.get("action", old_char.get("action"))
                old_char["natural_motion"] = new_char.get(
                    "natural_motion",
                    old_char.get("natural_motion", []),
                )
                old_char["uniqueness"] = new_char.get(
                    "uniqueness",
                    old_char.get("uniqueness"),
                )

        return scene

    @staticmethod
    def _build_prompt(user_prompt: str) -> str:

        return f"""
    Anda adalah Sutradara Film Animasi, Character Designer, Cinematographer, dan Environment Artist profesional.

    Balas HANYA JSON valid.
    Jangan markdown.
    Jangan komentar.
    Jangan penjelasan.
    Jangan <think>.
    Jangan ```json.

    Buat tepat 4 scene.

    Format:

    {{
    "title":"",
    "logline":"",
    "style":"",
    "scenes":[]
    }}

    Setiap scene berisi:

    title
    setting
    scene_action
    characters
    camera
    mood
    director_notes
    duration
    narration

    SETTING adalah object:

    location
    time
    weather
    season
    lighting
    environment[]
    environment_motion[]

    Environment harus kaya detail serta hidup.
    Contoh motion:
    waves rolling
    river flowing
    grass moving
    tree leaves swaying
    clouds drifting
    birds flying
    rain falling
    fire flickering
    smoke drifting

    CHARACTERS adalah array object.

    Setiap karakter memiliki:

    name
    role
    gender
    age
    identity
    appearance
    clothes
    expression
    action
    natural_motion
    uniqueness

    Karakter harus konsisten di seluruh scene.

    Jika terdapat beberapa karakter, setiap karakter wajib mempunyai identitas visual berbeda secara alami (umur, tinggi, bentuk tubuh, wajah, warna kulit, rambut, mata, pakaian, warna pakaian, aksesoris, postur, dll).

    Hanya jika cerita menyebut "kembar identik", buat penampilan mereka hampir sama.

    natural_motion berisi gerakan kecil alami sesuai kondisi.
    Contoh:
    eyes blinking
    breathing naturally
    hair moving with wind
    clothes moving softly
    head turning slightly
    hand gestures while talking
    body weight shifting
    walking naturally
    smiling naturally

    scene_action adalah daftar aksi utama yang benar-benar terlihat oleh kamera.

    camera adalah object:

    shot
    angle
    movement
    focus

    Gunakan variasi shot dan movement yang sinematik.

    mood adalah object:

    emotion
    atmosphere
    lighting
    color_tone

    director_notes maksimal 4 kalimat singkat yang membantu visual, misalnya:
    rich cinematic composition
    natural acting
    detailed environment
    consistent character appearance

    Jangan membuat image_prompt.
    Jangan membuat prompt Stable Diffusion.

    Ide cerita:

    {user_prompt}
    """.strip()

    @staticmethod
    def _extract_json(text: str) -> str:

        first = text.find("{")

        last = text.rfind("}")

        if first == -1 or last == -1:
            raise ValueError("JSON tidak ditemukan.")

        return text[first:last + 1]

    @staticmethod
    def _repair_json(text: str) -> str:

        text = text.strip()

        text = text.replace("```json", "")

        text = text.replace("```", "")

        return text

    def _parse_json(self, raw: str) -> dict:

        text = self._repair_json(raw)

        text = self._extract_json(text)

        return json.loads(text)


def normalize_storyboard(data: dict, user_prompt: str) -> Storyboard:
    if not isinstance(data, dict):
        raise ValueError("Storyboard AI tidak berbentuk objek JSON.")

    scenes_raw = data.get("scenes")
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise ValueError("Storyboard AI tidak memiliki adegan.")

    scenes=[]

    for index,item in enumerate(scenes_raw[:4],start=1):
        if not isinstance(item,dict):
            continue

        scene=Scene(
            number=index,
            title=str(item.get("title") or f"Adegan {index}"),
            duration=safe_float(item.get("duration"),3.0),

            setting=item.get("setting") if isinstance(item.get("setting"),dict) else {
                "location":"","time":"","weather":"","season":"",
                "lighting":"","environment":[],"environment_motion":[]
            },

            scene_action=item.get("scene_action") if isinstance(item.get("scene_action"),list) else [],

            characters=item.get("characters") if isinstance(item.get("characters"),list) else [],

            camera=item.get("camera") if isinstance(item.get("camera"),dict) else {
                "shot":"Medium Shot",
                "angle":"Eye Level",
                "movement":"Static",
                "focus":"Karakter Utama"
            },

            mood=item.get("mood") if isinstance(item.get("mood"),dict) else {
                "emotion":"Tenang",
                "atmosphere":"Natural",
                "lighting":"Soft",
                "color_tone":"Natural"
            },

            narration=str(item.get("narration") or ""),
            director_notes=item.get("director_notes") if isinstance(item.get("director_notes"),list) else [],
        )

        scenes.append(scene)

    if not scenes:
        raise ValueError("Adegan AI kosong.")

    return Storyboard(
        title=str(data.get("title") or "Animasi Lokal"),
        logline=str(data.get("logline") or user_prompt),
        style=str(data.get("style") or "Animated Feature Film"),
        scenes=scenes,
    )


def fallback_storyboard(user_prompt: str) -> Storyboard:

    title = "Animasi " + " ".join(user_prompt.split()[:4]).title()

    scenes = [
        Scene(
            number=1,
            title="Opening",
            narration=user_prompt,
            duration=3.5,
        ),
        Scene(
            number=2,
            title="Development",
            narration=user_prompt,
            duration=3.5,
        ),
        Scene(
            number=3,
            title="Climax",
            narration=user_prompt,
            duration=3.5,
        ),
        Scene(
            number=4,
            title="Ending",
            narration=user_prompt,
            duration=3.5,
        ),
    ]

    return Storyboard(
        title=title,
        logline=user_prompt,
        style="DreamWorks style 3D animated feature film",
        scenes=scenes,
    )


def fallback_scene_details(scene: Scene) -> Scene:

    scene.setting = {
        "location": "small town park",
        "time": "morning",
        "weather": "clear sky",
        "season": "spring",
        "lighting": "soft morning sunlight",
        "environment": [
            "large trees",
            "grass",
            "walking path",
            "flowers",
        ],
        "environment_motion": [
            "tree leaves swaying",
            "grass moving with wind",
            "birds flying",
        ],
    }

    scene.scene_action = [
        "walking naturally",
        "looking around",
    ]

    scene.characters = [
        {
            "name": "Tokoh Utama",
            "role": "main character",
            "gender": "unknown",
            "age": "young",
            "identity": "main protagonist",
            "appearance": "short black hair, brown eyes, expressive face",
            "clothes": "casual hoodie, jeans, sneakers",
            "expression": "curious",
            "action": "walking",
            "natural_motion": [
                "eyes blinking naturally",
                "breathing naturally",
                "hair moving softly",
                "subtle body movement",
            ],
            "uniqueness": "simple everyday appearance",
        }
    ]

    scene.camera = {
        "shot": "wide shot",
        "angle": "eye level",
        "movement": "slow dolly in",
        "focus": "main character",
    }

    scene.mood = {
        "emotion": "peaceful",
        "atmosphere": "calm",
        "lighting": "soft natural light",
        "color_tone": "warm green",
    }

    scene.director_notes = [
        "rich cinematic composition",
        "natural character acting",
        "detailed environment",
        "keep character appearance consistent",
    ]

    return scene

    return Storyboard(

        title=title,

        logline=user_prompt,

        style="DreamWorks style 3D animated feature film",

        scenes=scenes,

    )


class PromptBuilder:

    STYLE = (
        "masterpiece, best quality, ultra detailed, cinematic, "
        "3D animated feature film, DreamWorks style, Pixar quality, "
        "global illumination, volumetric lighting, physically based rendering"
    )

    NEGATIVE = (
        "low quality, blurry, worst quality, bad anatomy, "
        "deformed, watermark, logo, text, stickman, "
        "flat illustration, sketch, comic, low poly"
    )

    def build(self, storyboard: Storyboard, scene: Scene) -> tuple[str, str]:

        prompt=[]

        prompt.append(self.STYLE)

        prompt.append(self._setting(scene))
        prompt.append(self._characters(scene))
        prompt.append(self._scene(scene))
        prompt.append(self._camera(scene))
        prompt.append(self._mood(scene))
        prompt.append(self._notes(scene))

        return ", ".join(filter(None,prompt)),self.NEGATIVE

    def _setting(self,scene):
        s=scene.setting
        parts=[
            s["location"],
            s["time"],
            s["weather"],
            s["season"],
            s["lighting"]
        ]
        parts.extend(s["environment"])
        parts.extend(s["environment_motion"])
        return ", ".join(filter(None,parts))

    def _characters(self,scene):
        result=[]
        for c in scene.characters:
            part=[]
            part.append(c.get("appearance",""))
            part.append(c.get("emotion",""))
            part.append(c.get("action",""))
            part.extend(c.get("natural_motion",[]))
            result.append(", ".join(filter(None,part)))
        return ", ".join(result)

    def _scene(self,scene):
        return ", ".join(scene.scene_action)

    def _camera(self,scene):
        c=scene.camera
        return ", ".join(filter(None,[
            c["shot"],
            c["angle"],
            c["movement"],
            c["focus"]
        ]))

    def _mood(self,scene):
        m=scene.mood
        return ", ".join(filter(None,[
            m["emotion"],
            m["atmosphere"],
            m["lighting"],
            m["color_tone"]
        ]))

    def _notes(self,scene):
        return ", ".join(scene.director_notes)


class ProjectStore:
    def __init__(self, root: Path = OUTPUT_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create_project(self, title: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.root / f"{stamp}_{slugify(title)}"
        (path / "images").mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(parents=True, exist_ok=True)
        (path / "video").mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def save_storyboard(path: Path, storyboard: Storyboard) -> Path:
        target = path / "storyboard.json"
        target.write_text(json.dumps(storyboard.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


class DirectorPromptGenerator:

    STYLE = [
        "DreamWorks animated feature film",
        "Pixar quality",
        "cinematic 3D animation",
        "physically based rendering",
        "global illumination",
        "ultra detailed",
        "highly detailed environment",
        "masterpiece",
    ]

    @classmethod
    def build(cls, storyboard, scene):

        parts = []

        parts.extend(cls.STYLE)

        parts.append(storyboard.style)

        s = scene.setting

        parts.extend(filter(None, [
            s.get("location"),
            s.get("time"),
            s.get("weather"),
            s.get("season"),
            s.get("lighting"),
        ]))

        parts.extend(s.get("environment", []))
        parts.extend(s.get("environment_motion", []))

        for c in scene.characters:

            parts.extend(filter(None, [

                c.get("role"),
                c.get("gender"),
                c.get("age"),

                c.get("identity"),

                c.get("appearance"),

                c.get("clothes"),

                c.get("expression"),

                c.get("action"),

                c.get("uniqueness"),

            ]))

            parts.extend(c.get("natural_motion", []))

        parts.extend(scene.scene_action)

        cam = scene.camera

        parts.extend(filter(None, [

            cam.get("shot"),
            cam.get("angle"),
            cam.get("movement"),
            cam.get("focus"),

        ]))

        mood = scene.mood

        parts.extend(filter(None, [

            mood.get("emotion"),
            mood.get("atmosphere"),
            mood.get("lighting"),
            mood.get("color_tone"),

        ]))

        parts.extend(scene.director_notes)

        parts.extend([
            "natural human anatomy",
            "expressive face",
            "alive eyes",
            "natural pose",
            "dynamic composition",
            "rich background",
            "consistent character design",
            "no empty space",
        ])

        return cls.clean(parts)

    @staticmethod
    def clean(parts):

        result = []
        seen = set()

        for item in parts:

            if item is None:
                continue

            item = str(item).strip()

            if not item:
                continue

            key = item.lower()

            if key in seen:
                continue

            seen.add(key)

            result.append(item)

        return ", ".join(result)


class DiffusionImageMaker:

    _pipe = None

    negative_prompt = ", ".join([

        # Quality
        "worst quality",
        "low quality",
        "bad quality",
        "low resolution",
        "blurry",
        "out of focus",
        "soft image",
        "pixelated",
        "jpeg artifacts",
        "noise",
        "grain",

        # Anatomy
        "bad anatomy",
        "bad proportions",
        "deformed",
        "mutated",
        "malformed",
        "disfigured",
        "twisted body",
        "broken limbs",
        "long neck",
        "duplicate body",
        "duplicate face",
        "extra head",
        "extra arms",
        "extra legs",
        "extra hands",
        "extra fingers",
        "missing fingers",
        "fused fingers",
        "cropped body",
        "floating limbs",

        # Face
        "ugly face",
        "deformed face",
        "bad eyes",
        "cross eyed",
        "empty eyes",
        "lifeless eyes",
        "asymmetrical eyes",
        "bad mouth",
        "bad teeth",
        "duplicate face",

        # Character consistency
        "different hairstyle",
        "different clothes",
        "different face",
        "different identity",
        "random accessories",
        "age inconsistency",

        # Scene
        "empty background",
        "plain background",
        "minimal background",
        "missing environment",
        "floating objects",
        "wrong perspective",
        "bad composition",
        "poor lighting",
        "flat lighting",
        "oversaturated",
        "underexposed",
        "overexposed",

        # Rendering
        "unfinished render",
        "low poly",
        "cgi preview",
        "wireframe",
        "plastic toy",
        "rubber skin",
        "wax skin",

        # Style
        "anime",
        "manga",
        "comic",
        "2d illustration",
        "line art",
        "sketch",
        "painting",
        "watercolor",
        "cartoon drawing",
        "clipart",

        # Text
        "text",
        "subtitle",
        "caption",
        "logo",
        "watermark",
        "signature",
        "frame",
        "border",

    ])

    def __init__(self, width=768, height=512):

        self.width = width
        self.height = height

        if DiffusionImageMaker._pipe is None:

            print("Loading Stable Diffusion...")

            model_path = Path("models/diffusion/stable-diffusion-1.5-fp16")

            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )

            pipe.to("cpu")
            pipe.set_progress_bar_config(disable=True)

            DiffusionImageMaker._pipe = pipe

            print("Stable Diffusion Ready")

        self.pipe = DiffusionImageMaker._pipe

    def render_scene_image(
        self,
        storyboard,
        scene,
        target,
    ):

        prompt = DirectorPromptGenerator.build(
            storyboard,
            scene,
        )

        print("=" * 100)
        print(prompt)
        print("=" * 100)

        generator = torch.Generator("cpu").manual_seed(
            random.randint(0, 999999999)
        )

        with torch.no_grad():

            image = self.pipe(

                prompt=prompt,

                negative_prompt=self.NEGATIVE_PROMPT,

                width=self.width,
                height=self.height,

                num_inference_steps=40,

                guidance_scale=9,

                generator=generator,

            ).images[0]

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(target)

        return target

    def render_scene_frame(
        self,
        storyboard,
        scene,
        frame_index,
        total_frames,
        target,
    ):

        return self.render_scene_image(
            storyboard,
            scene,
            target,
        )


class VideoMaker:
    def __init__(self, fps: int = 12):
        self.fps = fps
        local_ffmpeg = LOCAL_TOOLS_DIR / "ffmpeg" / "bin" / "ffmpeg.exe"
        self.ffmpeg = str(local_ffmpeg) if local_ffmpeg.exists() else shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("FFmpeg tidak ditemukan di PATH.")

    def render_video(self, frames_dir: Path, target: Path) -> Path:
        pattern = frames_dir / "frame_%06d.png"
        command = [
            self.ffmpeg,
            "-y",
            "-framerate",
            str(self.fps),
            "-i",
            str(pattern),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "FFmpeg gagal membuat video.")
        return target


def sort_models_for_laptop(models: list[str]) -> list[str]:
    preferred = [
        "qwen2.5:0.5b",
        "qwen3:0.6b",
        "gemma3:1b",
        "qwen2.5:1.5b",
        "qwen3-4b-local:latest",
    ]
    rank = {name: index for index, name in enumerate(preferred)}
    return sorted(models, key=lambda name: (rank.get(name, 99), name.lower()))


class ArAnimatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1280x780")
        self.root.minsize(1040, 680)

        self.store = ProjectStore()
        self.messages: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.storyboard: Storyboard | None = None
        self.project_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.frame_paths: list[Path] = []
        self.video_path: Path | None = None
        self.preview_photo = None
        self.video_preview_photo = None
        self.busy = False
        self.fps_var = tk.IntVar(value=12)
        self.width_var = tk.IntVar(value=960)
        self.height_var = tk.IntVar(value=544)
        self.current_prompt = ""
        self.current_model = DEFAULT_LLM
        self.current_width = 960
        self.current_height = 544
        self.current_fps = 12

        self._build_style()
        self._build_ui()
        self._poll_messages()

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", padding=(12, 7), font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(side=TOP, fill=X)

        ttk.Label(top, text="Ide video animasi", style="Header.TLabel").pack(anchor="w")
        self.prompt_text = ScrolledText(top, height=4, wrap=tk.WORD, font=("Segoe UI", 10))
        self.prompt_text.pack(fill=X, pady=(6, 8))
        self.prompt_text.insert(
            END,
            "Seekor anak menemukan robot kecil di taman kota, lalu mereka bekerja sama "
            "menyalakan kembali lampu festival malam.",
        )

        controls = ttk.Frame(top)
        controls.pack(fill=X)
        ttk.Label(controls, text="Model GGUF").pack(side=LEFT)
        models = list_local_models() or [DEFAULT_LLM]
        self.model_var = tk.StringVar(value=models[0])
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, values=models, width=32)
        self.model_combo.pack(side=LEFT, padx=(8, 16))

        ttk.Label(controls, text="Ukuran").pack(side=LEFT)
        ttk.Combobox(controls, textvariable=self.width_var, values=[640, 856, 960, 1280], width=6, state="readonly").pack(side=LEFT, padx=(6, 2))
        ttk.Label(controls, text="x").pack(side=LEFT)
        ttk.Combobox(controls, textvariable=self.height_var, values=[360, 480, 544, 720], width=6, state="readonly").pack(side=LEFT, padx=(2, 12))

        ttk.Label(controls, text="FPS").pack(side=LEFT)
        ttk.Combobox(controls, textvariable=self.fps_var, values=[8, 12, 15, 24], width=5, state="readonly").pack(side=LEFT, padx=(6, 16))

        self.all_button = ttk.Button(controls, text="Buat Semua", style="Accent.TButton", command=self.make_all)
        self.all_button.pack(side=LEFT, padx=(0, 6))
        self.story_button = ttk.Button(controls, text="Storyboard", command=self.make_storyboard)
        self.story_button.pack(side=LEFT, padx=3)
        self.image_button = ttk.Button(controls, text="Gambar", command=self.make_images)
        self.image_button.pack(side=LEFT, padx=3)
        self.video_button = ttk.Button(controls, text="Video", command=self.make_video)
        self.video_button.pack(side=LEFT, padx=3)
        ttk.Button(controls, text="Buka Folder", command=self.open_project_folder).pack(side=RIGHT)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=BOTH, expand=True, padx=12, pady=(0, 12))

        left = ttk.PanedWindow(body, orient=tk.VERTICAL)
        body.add(left, weight=2)

        chat_frame = ttk.LabelFrame(left, text="Percakapan")
        self.chat_text = ScrolledText(chat_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        self.chat_text.pack(fill=BOTH, expand=True, padx=8, pady=8)
        self.chat_text.configure(state=DISABLED)
        left.add(chat_frame, weight=2)

        log_frame = ttk.LabelFrame(left, text="Proses Pembuatan")
        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 9), height=10)
        self.log_text.pack(fill=BOTH, expand=True, padx=8, pady=8)
        self.log_text.configure(state=DISABLED)
        left.add(log_frame, weight=1)

        right = ttk.Notebook(body)
        body.add(right, weight=3)

        image_tab = ttk.Frame(right, padding=10)
        right.add(image_tab, text="Layar Gambar")
        image_split = ttk.PanedWindow(image_tab, orient=tk.HORIZONTAL)
        image_split.pack(fill=BOTH, expand=True)
        list_frame = ttk.Frame(image_split)
        self.scene_list = tk.Listbox(list_frame, font=("Segoe UI", 10), activestyle="dotbox")
        self.scene_list.pack(fill=BOTH, expand=True)
        self.scene_list.bind("<<ListboxSelect>>", self.on_scene_select)
        image_split.add(list_frame, weight=1)
        self.image_label = ttk.Label(image_split, text="Gambar adegan akan tampil di sini.", anchor="center")
        image_split.add(self.image_label, weight=4)

        video_tab = ttk.Frame(right, padding=10)
        right.add(video_tab, text="Layar Video")
        self.video_label = ttk.Label(video_tab, text="Video akhir akan tampil di sini setelah render.", anchor="center")
        self.video_label.pack(fill=BOTH, expand=True)
        video_actions = ttk.Frame(video_tab)
        video_actions.pack(fill=X, pady=(8, 0))
        ttk.Button(video_actions, text="Putar Video", command=self.play_video).pack(side=LEFT)
        ttk.Button(video_actions, text="Pilih Folder Output", command=self.choose_output_folder).pack(side=LEFT, padx=6)
        self.video_path_label = ttk.Label(video_actions, text="Belum ada video.")
        self.video_path_label.pack(side=LEFT, padx=12)

        self._append_chat("AI", "Siap menjadi sutradara lokal. Tulis ide, lalu klik Buat Semua.")
        self._append_log("Aplikasi siap. Menggunakan Local LLM (GGUF) melalui llama-cpp.")

    def run_task(self, target, *args) -> None:
        if self.busy:
            messagebox.showinfo(APP_NAME, "Proses lain masih berjalan.")
            return
        self.current_prompt = self.prompt_text.get("1.0", END).strip()
        self.current_model = self.model_var.get()
        self.current_width = int(self.width_var.get())
        self.current_height = int(self.height_var.get())
        self.current_fps = int(self.fps_var.get())
        self.busy = True
        self._set_buttons(False)
        thread = threading.Thread(target=self._task_wrapper, args=(target, args), daemon=True)
        thread.start()

    def _task_wrapper(self, target, args) -> None:
        try:
            target(*args)
        except Exception as exc:
            self.messages.put(("error", str(exc)))
        finally:
            self.messages.put(("done", None))

    def make_storyboard(self) -> None:
        self.run_task(self._create_storyboard_task)

    def make_images(self) -> None:
        self.run_task(self._render_images_task)

    def make_video(self) -> None:
        self.run_task(self._render_video_task)

    def make_all(self) -> None:
        self.run_task(self._make_all_task)

    def _make_all_task(self) -> None:
        self._create_storyboard_task()
        self._render_images_task()
        self._render_video_task()

def _create_storyboard_task(self):

    try:

        self.messages.put(("log", "Memuat model AI lokal..."))

        llm = LocalLLM(self.current_model)

        storyboard = llm.create_storyboard(
            self.current_prompt
        )

        self.messages.put(("log", "Memperdalam setiap scene..."))

        total = len(storyboard.scenes)

        for i, scene in enumerate(storyboard.scenes, 1):

            self.messages.put(("log", f"Scene {i}/{total}"))

            detail = llm.expand_scene(
                storyboard,
                scene,
            )

            normalize_scene_details(
                scene,
                detail,
            )

        self.storyboard = storyboard

        self.project_dir = self.store.create_project(
            storyboard.title
        )

        self.messages.put(("storyboard", storyboard))

        self.messages.put((
            "chat",
            (
                "AI",
                f"Storyboard selesai dibuat.\n\nJudul : {storyboard.title}"
            )
        ))

        self.messages.put(("log", "Storyboard selesai."))

    except Exception:

        self.messages.put((
            "error",
            traceback.format_exc()
        ))

    def _render_images_task(self) -> None:
        self.messages.put(("log", "Mulai render gambar"))
        if not self.storyboard or not self.project_dir:
            raise ValueError("Buat storyboard terlebih dahulu.")
        self.messages.put(("log", "Membuat ImageMaker"))    
        maker = DiffusionImageMaker(width=self.current_width,height=self.current_height,)
        image_dir = self.project_dir / "images"
        self.image_paths = []
        for scene in self.storyboard.scenes:
            self.messages.put(("log", f"Render scene {scene.number}"))
            target = image_dir / f"scene_{scene.number:03d}.png"
            maker.render_scene_image(self.storyboard, scene, target)
            self.image_paths.append(target)
            self.messages.put(("image", target))
            self.messages.put(("log", f"Gambar adegan {scene.number} selesai: {target.name}"))
        self.messages.put(("log", "Semua gambar adegan selesai."))

    def _render_video_task(self) -> None:
        if not self.storyboard or not self.project_dir:
            raise ValueError("Buat storyboard terlebih dahulu.")
        maker = DiffusionImageMaker(width=self.current_width, height=self.current_height)
        fps = self.current_fps
        frames_dir = self.project_dir / "frames"
        for existing in frames_dir.glob("frame_*.png"):
            existing.unlink()

        frame_number = 1
        self.frame_paths = []
        for scene in self.storyboard.scenes:
            total = max(1, int(scene.duration * fps))
            self.messages.put(("log", f"Render frame adegan {scene.number}: {total} frame"))
            for local_index in range(total):
                target = frames_dir / f"frame_{frame_number:06d}.png"
                maker.render_scene_frame(self.storyboard, scene, local_index, total, target)
                self.frame_paths.append(target)
                frame_number += 1
                if local_index == 0:
                    self.messages.put(("video_preview", target))

        target = self.project_dir / "video" / f"{slugify(self.storyboard.title)}.mp4"
        video = VideoMaker(fps=fps).render_video(frames_dir, target)
        self.video_path = video
        self.messages.put(("video", video))
        self.messages.put(("log", f"Video selesai: {video}"))

    def _poll_messages(self) -> None:
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "chat":
                who, text = payload
                self._append_chat(who, text)
            elif kind == "log":
                self._append_log(str(payload))
            elif kind == "storyboard":
                self._show_storyboard(payload)
            elif kind == "image":
                self._add_image_path(Path(payload))
            elif kind == "video_preview":
                self._show_video_preview(Path(payload))
            elif kind == "video":
                self._show_video(Path(payload))
            elif kind == "error":
                self._append_log(f"ERROR: {payload}")
                messagebox.showerror(APP_NAME, str(payload))
            elif kind == "done":
                self.busy = False
                self._set_buttons(True)
        self.root.after(120, self._poll_messages)

    def _set_buttons(self, enabled: bool) -> None:
        state = NORMAL if enabled else DISABLED
        for button in [self.all_button, self.story_button, self.image_button, self.video_button]:
            button.configure(state=state)

    def _append_chat(self, who: str, text: str) -> None:
        self.chat_text.configure(state=NORMAL)
        self.chat_text.insert(END, f"{who}:\n{text}\n\n")
        self.chat_text.see(END)
        self.chat_text.configure(state=DISABLED)

    def _append_log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=NORMAL)
        self.log_text.insert(END, f"[{stamp}] {text}\n")
        self.log_text.see(END)
        self.log_text.configure(state=DISABLED)

    def _show_storyboard(self, storyboard: Storyboard) -> None:
        self.scene_list.delete(0, END)
        for scene in storyboard.scenes:
            self.scene_list.insert(END, f"{scene.number}. {scene.title} ({scene.duration:g}s)")

    def _add_image_path(self, path: Path) -> None:
        if path not in self.image_paths:
            self.image_paths.append(path)
        self._show_image(path)

    def _show_image(self, path: Path) -> None:
        if ImageTk is None:
            self.image_label.configure(text=str(path))
            return
        image = Image.open(path).convert("RGB")
        image.thumbnail((760, 460))
        self.preview_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_photo, text="")

    def _show_video_preview(self, path: Path) -> None:
        if ImageTk is None:
            self.video_label.configure(text=str(path))
            return
        image = Image.open(path).convert("RGB")
        image.thumbnail((760, 460))
        self.video_preview_photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.video_preview_photo, text="")

    def _show_video(self, path: Path) -> None:
        self.video_path_label.configure(text=str(path))
        self._append_chat("AI", f"Video selesai dibuat:\n{path}")
        if self.frame_paths:
            self._show_video_preview(self.frame_paths[0])

    def on_scene_select(self, _event=None) -> None:
        selection = self.scene_list.curselection()
        if not selection or not self.project_dir:
            return
        index = selection[0]
        candidate = self.project_dir / "images" / f"scene_{index + 1:03d}.png"
        if candidate.exists():
            self._show_image(candidate)

    def open_project_folder(self) -> None:
        path = self.project_dir or OUTPUT_DIR
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    def choose_output_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(OUTPUT_DIR))
        if selected:
            self.project_dir = Path(selected)
            self._append_log(f"Folder proyek aktif: {self.project_dir}")

    def play_video(self) -> None:
        if self.video_path and self.video_path.exists():
            os.startfile(self.video_path)
        else:
            messagebox.showinfo(APP_NAME, "Belum ada video yang bisa diputar.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DIFFUSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> int:
    if Image is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            APP_NAME,
            "Pillow belum terpasang.\n\nJalankan install.bat terlebih dahulu, lalu buka run.bat.",
        )
        return 1
    root = tk.Tk()
    ArAnimatorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
