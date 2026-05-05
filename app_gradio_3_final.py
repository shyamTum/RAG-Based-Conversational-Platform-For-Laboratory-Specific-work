import os
import json
import time
import html
from typing import List, Dict, Any

import numpy as np
import faiss
import gradio as gr
import requests
from sentence_transformers import SentenceTransformer

KB_DIR = r".\kb_store"
MODEL_DIR = r".\sentence-transformer\sentence-transformer\all-MiniLM-L6-v2"

MODEL_CONFIGS = {
    "Gemma": {
        "base_url": "http://MB-QS-PP-H200.int.pg.com:8842/v1",
        "model": "google/gemma-3-27b-it",
    },
    "Pixtral": {
        "base_url": "http://MB-QS-PP-H200.int.pg.com:8844/v1",
        "model": "mistralai/Pixtral-12B-2409",
    },
    "Qwen3-VL": {
        "base_url": "http://MB-QS-PP-H200.int.pg.com:8841/v1",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
    },
    "InternVL3.5": {
        "base_url": "http://MB-QS-PP-H200.int.pg.com:8843/v1",
        "model": "OpenGVLab/InternVL3_5-14B",
    },
}

LLM_API_KEY = os.getenv("LLM_API_KEY", "")

MAX_CONTEXT_CHARS = 12000
TIMEOUT_SEC = 180
RETRIES = 2


def paths():
    return {
        "index": os.path.join(KB_DIR, "faiss.index"),
        "meta": os.path.join(KB_DIR, "meta.jsonl"),
    }


def load_meta() -> List[Dict[str, Any]]:
    rows = []
    with open(paths()["meta"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_index() -> faiss.Index:
    return faiss.read_index(paths()["index"])


def embed_query(embedder: SentenceTransformer, q: str) -> np.ndarray:
    v = embedder.encode(
        [q],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(v)
    return v


def retrieve(
    index: faiss.Index,
    meta_rows: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    query: str,
    k: int
):
    qv = embed_query(embedder, query)
    scores, idxs = index.search(qv, k)

    hits = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if 0 <= idx < len(meta_rows):
            row = dict(meta_rows[idx])
            row["score"] = float(score)
            hits.append(row)
    return hits


def build_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    total = 0

    for h in hits:
        src = h.get("relative_path", h.get("source", "unknown"))
        cid = h.get("chunk_id", -1)
        txt = (h.get("text") or "").strip()

        block = f"[SOURCE: {src} | chunk {cid}]\n{txt}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total += len(block)

    return "\n---\n".join(blocks)


# def call_llm(
#     llm_choice: str,
#     question: str,
#     context: str,
#     temperature: float,
#     max_tokens: int
# ) -> str:
#     config = MODEL_CONFIGS[llm_choice]
#     llm_base_url = config["base_url"]
#     llm_model = config["model"]

#     url = llm_base_url.rstrip("/") + "/chat/completions"

#     headers = {
#         "Content-Type": "application/json",
#     }
#     if LLM_API_KEY.strip():
#         headers["Authorization"] = f"Bearer {LLM_API_KEY}"

#     system = (
#         "You are a helpful assistant for question answering over retrieved documents. "
#         "Answer directly and completely. Avoid unnecessary introductions. "
#         "Use the retrieved context first. If the context is incomplete, briefly say so and then continue with general knowledge. "
#         "If the user asks for equations or mathematical expressions, include them whenever possible, but only if asked. "
#         "If the user asks for a diagram or flowchart, provide a simple text-based diagram or ASCII flowchart, but only if asked. "
#         "Do not stop early. Make sure all major parts of the user's request are addressed."
#     )

#     user = (
#         f"CONTEXT:\n{context}\n\n"
#         f"QUESTION:\n{question}\n\n"
#         "Instructions:\n"
#         "- Give a complete answer.\n"
#         "- If the user requests a minimum length, satisfy it.\n"
#         "- If mathematical expressions are requested, include them.\n"
#         "- If a diagram or flowchart is requested, include a compact text-based one only if requested.\n"
#         "- If retrieved context is incomplete, state that briefly and then continue with general knowledge only if needed.\n"
#         "- Do not end mid-answer."
#     )

#     payload = {
#         "model": llm_model,
#         "messages": [
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ],
#         "temperature": float(temperature),
#         "max_tokens": int(max_tokens),
#     }

#     last_err = None

#     for attempt in range(RETRIES + 1):
#         try:
#             r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)

#             if r.status_code != 200:
#                 return f"LLM error ({llm_choice}) {r.status_code}: {r.text}"

#             data = r.json()
#             return data["choices"][0]["message"]["content"]

#         except requests.exceptions.ReadTimeout:
#             last_err = f"ReadTimeout after {TIMEOUT_SEC}s (attempt {attempt + 1}/{RETRIES + 1})"
#         except Exception as e:
#             last_err = f"Error calling {llm_choice}: {e}"

#         time.sleep(1.5 * (attempt + 1))

#     return last_err or f"Unknown error calling {llm_choice}."

def call_llm(
    llm_choice: str,
    question: str,
    context: str,
    temperature: float,
    max_tokens: int
) -> str:
    config = MODEL_CONFIGS[llm_choice]
    llm_base_url = config["base_url"]
    llm_model = config["model"]

    url = llm_base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }
    if LLM_API_KEY.strip():
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    system = (
        "You are a helpful assistant for question answering over retrieved documents. "
        "Answer directly and completely. Avoid unnecessary introductions. "
        "Use the retrieved context first. If the context is incomplete, briefly say so and then continue with general knowledge. "
        "If the user asks for equations or mathematical expressions, include them whenever possible, but only if asked. "
        "If the user asks for a diagram or flowchart, provide a simple text-based diagram or ASCII flowchart, but only if asked. "
        "Prefer a compact but complete answer. Avoid unnecessarily long numbered lists unless they help clarity. "
        "Do not stop early. Make sure all major parts of the user's request are addressed."
    )

    user = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Instructions:\n"
        "- Give a complete answer.\n"
        "- If the user requests a minimum length, satisfy it.\n"
        "- If mathematical expressions are requested, include them.\n"
        "- If a diagram or flowchart is requested, include a compact text-based one only if requested.\n"
        "- If retrieved context is incomplete, state that briefly and then continue with general knowledge only if needed.\n"
        "- Avoid unnecessary wording.\n"
        "- Do not end mid-answer."
    )

    def single_call(messages):
        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)

        if r.status_code != 200:
            return None, f"LLM error ({llm_choice}) {r.status_code}: {r.text}", None

        data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "")
        return text, None, finish_reason

    last_err = None

    for attempt in range(RETRIES + 1):
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

            full_answer_parts = []
            max_continuations = 5   # you can increase to 6 or 7 if needed

            for step in range(max_continuations):
                text, err, finish_reason = single_call(messages)

                if err:
                    if full_answer_parts:
                        return "".join(full_answer_parts) + f"\n\n[Continuation failed: {err}]"
                    return err

                full_answer_parts.append(text)

                if finish_reason != "length":
                    return "".join(full_answer_parts)

                # Ask model to continue from exactly where it stopped
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Continue exactly from where you stopped. "
                        "Do not repeat earlier text. "
                        "Finish the answer completely."
                    )
                })

            return "".join(full_answer_parts) + "\n\n[Answer may still be truncated after multiple continuations.]"

        except requests.exceptions.ReadTimeout:
            last_err = f"ReadTimeout after {TIMEOUT_SEC}s (attempt {attempt + 1}/{RETRIES + 1})"
        except Exception as e:
            last_err = f"Error calling {llm_choice}: {e}"

        time.sleep(1.5 * (attempt + 1))

    return last_err or f"Unknown error calling {llm_choice}."


def build_sources_html_and_choices(hits: List[Dict[str, Any]]):
    sources_html = "<ul>"
    choices = []

    for h in hits:
        full_path = h.get("path", "")
        rel = h.get("relative_path", h.get("source", "unknown"))
        score = h.get("score", 0.0)
        chunk = h.get("chunk_id", -1)

        label = f"{rel} (chunk {chunk}, score={score:.3f})"
        choices.append((label, full_path))

        safe_label = html.escape(label)
        sources_html += f"<li>{safe_label}</li>"

    sources_html += "</ul>"
    return sources_html, choices


def qa(llm_choice: str, question: str, top_k: int, temperature: float, max_tokens: int):
    if not question.strip():
        return "Please enter a question.", "", gr.update(choices=[], value=None), ""

    if llm_choice not in MODEL_CONFIGS:
        return f"Unknown model selection: {llm_choice}", "", gr.update(choices=[], value=None), ""

    if not os.path.isdir(MODEL_DIR):
        return f"MODEL_DIR not found: {MODEL_DIR}", "", gr.update(choices=[], value=None), ""

    if not os.path.exists(paths()["index"]) or not os.path.exists(paths()["meta"]):
        return "KB not found. Run: python ingest_kb_3.py", "", gr.update(choices=[], value=None), ""

    embedder = SentenceTransformer(MODEL_DIR)
    index = load_index()
    meta_rows = load_meta()

    hits = retrieve(index, meta_rows, embedder, question, int(top_k))
    context = build_context(hits)
    sources_html, source_choices = build_sources_html_and_choices(hits)

    if not context.strip():
        return "No relevant chunks retrieved. Try increasing top-k.", sources_html, gr.update(choices=source_choices, value=None), ""

    ans = call_llm(llm_choice, question, context, float(temperature), int(max_tokens))
    return ans, sources_html, gr.update(choices=source_choices, value=None), ""


def open_selected_folder(selected_path: str):
    if not selected_path:
        return "Please select a source first."

    try:
        folder_path = os.path.dirname(selected_path)
        os.startfile(folder_path)  # Windows
        return f"Opened folder: {folder_path}"
    except Exception as e:
        return f"Could not open folder: {e}"


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            model_choice = gr.Dropdown(
                choices=["Gemma", "Pixtral", "Qwen3-VL", "InternVL3.5"],
                value="Gemma",
                label="Model"
            )

            question = gr.Textbox(label="Question", lines=2)

            top_k = gr.Slider(
                1, 20, value=6, step=1,
                label="Top-k",
                info="How many relevant document chunks to search and use for answering."
            )

            temperature = gr.Slider(
                0.0, 1.0, value=0.2, step=0.05,
                label="Temperature",
                info="Controls creativity: lower = more factual, higher = more creative."
            )

            max_tokens = gr.Slider(
                128, 2048, value=768, step=32,
                label="Max tokens",
                info="Maximum length of the answer. Higher = longer responses."
            )

            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column():
            #answer = gr.Textbox(label="Answer", lines=18)
            answer = gr.Textbox(label="Answer")
            sources = gr.HTML(label="Sources")
            source_selector = gr.Dropdown(label="Select Source to Open", choices=[], value=None)
            open_btn = gr.Button("Open Selected Folder")
            open_status = gr.Textbox(label="Open Status", interactive=False)

    submit_btn.click(
        fn=qa,
        inputs=[model_choice, question, top_k, temperature, max_tokens],
        outputs=[answer, sources, source_selector, open_status],
        show_progress="minimal"
    )

    open_btn.click(
        fn=open_selected_folder,
        inputs=[source_selector],
        outputs=[open_status],
        show_progress="hidden"
    )

    clear_btn.click(
        fn=lambda: ("", "", gr.update(choices=[], value=None), "", ""),
        inputs=[],
        outputs=[question, answer, source_selector, open_status, sources],
        show_progress="hidden"
    )

demo.launch(server_name="127.0.0.1", share=False)