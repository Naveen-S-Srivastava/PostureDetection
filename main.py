"""
main.py - Medhya (NeuroPath) Personal Doctor + Posture API (final)

Endpoints:
 - POST /chat    -> general personal doctor assistant (concise, safe)
 - POST /posture -> upload image -> YOLOv8-pose -> posture metrics -> annotated image + LLM advice

Config:
 - Put OPENROUTER_API_KEY=sk-... in .env
 - Set POSE_MODEL_PATH to your best.pt absolute path
"""

import os
import math
import json
import time
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------
# Config - edit these only
# ---------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "meta-llama/llama-3.3-70b-instruct:free"  # change if needed

# Point this to your trained pose .pt (absolute path)
POSE_MODEL_PATH = "model/best.pt"

# Output directory (annotated images)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# sanity checks
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing in .env")

if not os.path.isfile(POSE_MODEL_PATH):
    raise RuntimeError(f"POSE_MODEL_PATH not found: {POSE_MODEL_PATH}")

# load pose model (may take a few seconds)
pose_model = YOLO(POSE_MODEL_PATH)

app = FastAPI(title="Medhya - Personal Doctor & Posture API", version="1.0")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# ---------------------------
# Geometry & posture utilities
# ---------------------------
def angle_degrees(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def normalized_to_pixels(arr_xy, w, h):
    arr = np.array(arr_xy, dtype=float)
    if arr.max() <= 1.0:
        arr[:, 0] *= w
        arr[:, 1] *= h
    return arr


def compute_posture_from_4kpts(kpts_xy, image_shape):
    h, w = image_shape[:2]
    pts = normalized_to_pixels(np.array(kpts_xy), w, h)

    ys = pts[:, 1]
    top_idx = int(np.argmin(ys))
    bot_idx = int(np.argmax(ys))
    mid_idxs = [i for i in range(len(pts)) if i not in (top_idx, bot_idx)]

    head = pts[top_idx]
    hip = pts[bot_idx]

    # chest/shoulder proxy selection
    if len(mid_idxs) >= 2:
        chest_idx = mid_idxs[0] if pts[mid_idxs[0], 1] < pts[mid_idxs[1], 1] else mid_idxs[1]
        other_idx = mid_idxs[1] if chest_idx == mid_idxs[0] else mid_idxs[0]
        chest = pts[chest_idx]
        mid_lower = pts[other_idx]
    elif len(mid_idxs) == 1:
        chest = pts[mid_idxs[0]]
        mid_lower = chest
    else:
        chest = (head + hip) / 2.0
        mid_lower = chest

    spine_angle = abs(angle_degrees(head, hip))  # 90 = vertical
    a1 = angle_degrees(head, chest)
    a2 = angle_degrees(chest, hip)
    neck_angle = abs(a1 - a2)
    neck_angle = abs(((neck_angle + 180) % 360) - 180)

    head_forward_norm = (head[0] - chest[0]) / w
    chest_hip_offset = (chest[0] - hip[0]) / w

    forward_head_flag = abs(head_forward_norm) > 0.03
    neck_slouch_flag = neck_angle > 15
    spine_verticalness = 90 - spine_angle
    spine_slouch_flag = spine_verticalness > 12
    rounded_shoulders_flag = abs(chest_hip_offset) > 0.05

    severity_score = 0
    severity_score += 2 if forward_head_flag else 0
    severity_score += 2 if neck_slouch_flag else 0
    severity_score += 1 if spine_slouch_flag else 0
    severity_score += 1 if rounded_shoulders_flag else 0

    if severity_score >= 5:
        label = "Severe slouch"
    elif severity_score >= 3:
        label = "Slouching"
    elif severity_score >= 1:
        label = "Mild slouch"
    else:
        label = "Good posture"

    return {
        "head": [float(head[0]), float(head[1])],
        "shoulders": [float(chest[0]), float(chest[1])],  # using 'shoulders' label per request
        "hips": [float(hip[0]), float(hip[1])],
        "spine_angle_deg": float(spine_angle),
        "neck_angle_deg": float(neck_angle),
        "head_forward_norm": float(head_forward_norm),
        "chest_hip_offset_norm": float(chest_hip_offset),
        "flags": {
            "forward_head": bool(forward_head_flag),
            "neck_slouch": bool(neck_slouch_flag),
            "spine_slouch": bool(spine_slouch_flag),
            "rounded_shoulders": bool(rounded_shoulders_flag),
        },
        "posture_label": label,
        "severity_score": severity_score,
    }


# ---------------------------
# Image drawing (clean modern, no sidebar)
# ---------------------------
def draw_clean_overlay(image_path: str, kpts_xy: np.ndarray) -> str:
    """
    Draws a clean modern overlay:
    - smooth blue spine (outer shadow + inner line)
    - colored keypoints: head red, shoulders green, hips yellow
    - black rounded label boxes with white text
    - no sidebar
    - returns saved filename under OUTPUT_DIR
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image for drawing")

    h, w = img.shape[:2]
    pts = normalized_to_pixels(np.array(kpts_xy), w, h).astype(int)

    # color definitions (BGR)
    spine_inner = (200, 80, 10)   # bluish tone
    spine_outer = (20, 20, 20)    # shadow
    head_color = (0, 0, 255)      # red
    shoulder_color = (0, 200, 0)  # green
    hip_color = (0, 215, 255)     # yellow

    # order top->...->bottom
    ordered = pts[np.argsort(pts[:, 1])]
    # smooth spine via polynomial interpolation if possible
    idxs = np.arange(len(ordered))
    spine_pts = None
    if len(ordered) >= 2:
        try:
            t = idxs
            t_dense = np.linspace(t.min(), t.max(), max(80, len(ordered) * 30))
            deg = min(len(ordered) - 1, 3)
            coeff_x = np.polyfit(t, ordered[:, 0], deg)
            coeff_y = np.polyfit(t, ordered[:, 1], deg)
            poly_x = np.polyval(coeff_x, t_dense).astype(int)
            poly_y = np.polyval(coeff_y, t_dense).astype(int)
            spine_pts = np.stack([poly_x, poly_y], axis=1)
            # outer stroke
            cv2.polylines(img, [spine_pts], isClosed=False, color=spine_outer, thickness=10, lineType=cv2.LINE_AA)
            # inner line
            cv2.polylines(img, [spine_pts], isClosed=False, color=spine_inner, thickness=4, lineType=cv2.LINE_AA)
        except Exception:
            spine_pts = ordered
            cv2.polylines(img, [ordered], isClosed=False, color=spine_inner, thickness=3, lineType=cv2.LINE_AA)
    else:
        # not enough points, skip smoothing
        spine_pts = ordered

    # pick top/mid/bot mapping for labels HEAD / SHOULDERS / HIPS
    top_idx = int(np.argmin(pts[:, 1]))
    bot_idx = int(np.argmax(pts[:, 1]))
    mid_idxs = [i for i in range(len(pts)) if i not in (top_idx, bot_idx)]
    head_pt = tuple(pts[top_idx])
    hip_pt = tuple(pts[bot_idx])
    if len(mid_idxs) >= 1:
        # choose shoulders as the higher middle point
        shoulder_idx = mid_idxs[0] if pts[mid_idxs[0], 1] < (pts[mid_idxs[1], 1] if len(mid_idxs) > 1 else 1e9) else mid_idxs[1] if len(mid_idxs) > 1 else mid_idxs[0]
        shoulder_pt = tuple(pts[shoulder_idx])
    else:
        shoulder_pt = tuple(((head_pt[0] + hip_pt[0]) // 2, (head_pt[1] + hip_pt[1]) // 2))

    # draw keypoints
    cv2.circle(img, head_pt, 9, head_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, shoulder_pt, 8, shoulder_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, hip_pt, 8, hip_color, -1, lineType=cv2.LINE_AA)

    # function to put label with rounded black box and white text
    font = cv2.FONT_HERSHEY_SIMPLEX
    def put_label(text, pos):
        x, y = pos
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        pad_x, pad_y = 8, 6
        rx, ry = x + 12, y - 12 - th
        # ensure within image
        rx = min(max(4, rx), w - tw - pad_x - 4)
        ry = min(max(4 + th, ry), h - 4)
        cv2.rectangle(img, (rx, ry - th - pad_y//2), (rx + tw + pad_x, ry + pad_y//2), (0,0,0), -1, lineType=cv2.LINE_AA)
        cv2.putText(img, text, (rx + 6, ry), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

    put_label("HEAD", (head_pt[0], head_pt[1]))
    put_label("SHOULDERS", (shoulder_pt[0], shoulder_pt[1]))
    put_label("HIPS", (hip_pt[0], hip_pt[1]))

    # top-left brand small banner - clean modern
    label_text = "Medhya posture"
    (tw, th), _ = cv2.getTextSize(label_text, font, 0.6, 1)
    cv2.rectangle(img, (12, 10), (18 + tw + 14, 10 + th + 10), (255,255,255), -1)
    cv2.putText(img, label_text, (18, 10 + th + 2), font, 0.6, (30,30,30), 1, cv2.LINE_AA)

    # save result
    filename = f"medhya_{int(time.time()*1000)}.jpg"
    out_path = OUTPUT_DIR / filename
    cv2.imwrite(str(out_path), img)
    return str(out_path.name)


# ---------------------------
# OpenRouter system prompt (tone: warm + clean modern)
# ---------------------------
SYSTEM_PROMPT = (
    "You are Medhya — a private personal doctor assistant. Speak like a warm, caring clinician with clear modern language. "
    "Be concise, supportive, and practical. Follow these rules:\n"
    "1) Return output in TWO parts: (A) a JSON object EXACTLY with keys: problems (list), severity (low|moderate|high), "
    "first_aid (list of 3 concise steps), exercises (list of 2 exercises), disclaimer (single sentence). "
    "Then a blank line, then (B) a short medium-length, point-wise patient message (3-5 bullets) in warm-clean tone. "
    "2) Lead with immediate actions if severity is moderate/high. 3) Avoid meta labels like 'Human explanation' or 'JSON'. "
    "4) Append one-line disclaimer: 'This is educational information, not a medical diagnosis. See a clinician if symptoms worsen.'"
)


def call_openrouter(prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> dict:
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenRouter error {resp.status_code}: {resp.text}")

    body = resp.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter malformed response: {e} / {body}")

    return {"raw": content, "parsed": None}


# ---------------------------
# API models & endpoints
# ---------------------------
class ChatRequest(BaseModel):
    user_message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def home():
    return {"message": "Medhya API running. Use /chat (POST) or /posture (POST file)."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_msg = request.user_message.strip()
    if not user_msg:
        return ChatResponse(reply="Hi — I'm Medhya, your personal doctor. How can I help today?")

    prompt = f"User question:\n{user_msg}\n\nReply concisely (2-6 short sentences) in a warm, clinical, modern tone. Include the one-line disclaimer."
    resp = call_openrouter(prompt, max_tokens=200, temperature=0.0)
    reply = resp.get("raw", "")
    # ensure we don't return the JSON-first format here; just return the text directly
    return ChatResponse(reply=reply.strip())


@app.post("/posture")
async def posture_endpoint(file: UploadFile = File(...)):
    # save tmp
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.flush()
        tmp.close()
        img_path = tmp.name

        # pose inference
        results = pose_model(img_path, conf=0.2, device="cpu")
        if len(results) == 0:
            return JSONResponse({"error": "no result from model"}, status_code=400)

        r = results[0]
        kpts_list = None
        try:
            if getattr(r, "keypoints", None) is not None and getattr(r.keypoints, "data", None) is not None:
                kpts_arr = np.array(r.keypoints.data)
                if kpts_arr.size == 0:
                    raise ValueError("empty keypoints")
                kpts = kpts_arr[0][:, :2]
                kpts_list = kpts
            elif getattr(r, "boxes", None) is not None and getattr(r.boxes, "keypoints", None) is not None:
                kpts_arr = np.array(r.boxes.keypoints)
                kpts = kpts_arr[0][:, :2]
                kpts_list = kpts
        except Exception:
            kpts_list = None

        if kpts_list is None:
            return JSONResponse({"error": "no keypoints detected"}, status_code=400)

        img = cv2.imread(img_path)
        if img is None:
            return JSONResponse({"error": "could not read uploaded image"}, status_code=400)

        metrics = compute_posture_from_4kpts(np.array(kpts_list), img.shape)

        # draw clean overlay and save
        saved_name = draw_clean_overlay(img_path, np.array(kpts_list))
        image_url = f"/output/{saved_name}"

        # build prompt asking for strict JSON then human text (no artifacts)
        llm_prompt = (
            f"Posture metrics:\n"
            f"- posture_label: {metrics['posture_label']}\n"
            f"- spine_angle_deg: {metrics['spine_angle_deg']:.1f}\n"
            f"- neck_angle_deg: {metrics['neck_angle_deg']:.1f}\n"
            f"- forward_head: {metrics['flags']['forward_head']}\n"
            f"- rounded_shoulders: {metrics['flags']['rounded_shoulders']}\n\n"
            "Return output as: FIRST a valid JSON object (one line or multi-line) with keys: problems (list), severity (low|moderate|high), "
            "first_aid (list of 3 short actionable steps), exercises (list of 2 short exercises), disclaimer (single sentence). "
            "Then a blank line, then a 3-5 bullet warm+clean patient-facing message (each bullet short). "
            "Do NOT output any extra meta labels. Keep JSON values short and clinical."
        )

        llm_resp = call_openrouter(llm_prompt, max_tokens=350, temperature=0.0)
        raw = llm_resp.get("raw", "")

        # attempt to extract JSON block then human text (cleanly)
        json_obj = None
        human_text = ""
        try:
            start = raw.find("{")
            end = raw.find("}", start)  # find first closing brace (may be naive)
            # better: find the matching closing brace for the first '{'
            if start != -1:
                depth = 0
                end_idx = -1
                for i in range(start, len(raw)):
                    if raw[i] == "{":
                        depth += 1
                    elif raw[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end_idx = i
                            break
                if end_idx != -1:
                    candidate = raw[start:end_idx+1]
                    json_obj = json.loads(candidate)
                    human_text = raw[end_idx+1:].strip()
                else:
                    # fallback: everything after first closing brace char
                    human_text = raw[start+1:].strip()
            else:
                human_text = raw.strip()
        except Exception:
            # if parsing fails, fallback: place entire raw into advice_text
            json_obj = None
            human_text = raw.strip()

        if json_obj is None:
            advice = {"advice_text": raw.strip()}
        else:
            advice = json_obj

        # ensure analysis_text is warm+clean bullets; if empty, set from raw
        analysis_text = human_text if human_text else (advice.get("advice_text") if isinstance(advice, dict) else "")

        response = {
            "posture_metrics": metrics,
            "first_aid": advice,
            "analysis_text": analysis_text,
            "image_url": image_url,
        }

        return JSONResponse(response)

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
