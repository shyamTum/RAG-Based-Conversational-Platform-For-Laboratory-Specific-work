import os
import json
import time
from typing import List, Dict, Any, Tuple

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


def retrieve(index, meta_rows, embedder, query: str, k: int):
    qv = embed_query(embedder, query)
    scores, idxs = index.search(qv, int(k))

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


def build_source_choices(hits: List[Dict[str, Any]]):
    choices = []

    for h in hits:
        full_path = h.get("path", "")
        rel = h.get("relative_path", h.get("source", "unknown"))
        score = h.get("score", 0.0)
        chunk = h.get("chunk_id", -1)

        label = f"{rel} (chunk {chunk}, score={score:.3f})"
        choices.append((label, full_path))

    return choices


def open_selected_folder(selected_path: str):
    if not selected_path:
        return

    try:
        folder_path = os.path.dirname(selected_path)
        os.startfile(folder_path)
    except Exception:
        pass


def openai_style_call(
    llm_choice: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, str]:
    config = MODEL_CONFIGS[llm_choice]
    url = config["base_url"].rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY.strip():
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)

    if r.status_code != 200:
        return f"LLM error ({llm_choice}) {r.status_code}: {r.text}", "error"

    data = r.json()
    text = data["choices"][0]["message"]["content"]
    finish_reason = data["choices"][0].get("finish_reason", "")

    return text, finish_reason


def call_llm(
    llm_choice: str,
    question: str,
    context: str,
    temperature: float,
    max_tokens: int,
) -> str:
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_err = None

    for attempt in range(RETRIES + 1):
        try:
            full_answer_parts = []
            max_continuations = 5

            for _ in range(max_continuations):
                text, finish_reason = openai_style_call(
                    llm_choice=llm_choice,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if finish_reason == "error":
                    return text

                full_answer_parts.append(text)

                if finish_reason != "length":
                    return "".join(full_answer_parts)

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


def history_for_rewrite(chat_history):
    pairs = []
    current_user = None

    for msg in chat_history or []:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            current_user = content
        elif role == "assistant" and current_user is not None:
            pairs.append((current_user, content))
            current_user = None

    return pairs


def format_recent_history(pair_history, history_window: int) -> str:
    if not pair_history or int(history_window) <= 0:
        return ""

    recent = pair_history[-int(history_window):]
    lines = []

    for user_msg, assistant_msg in recent:
        if user_msg:
            lines.append(f"User: {user_msg}")
        if assistant_msg:
            lines.append(f"Assistant: {assistant_msg[:800]}")

    return "\n".join(lines)


def rewrite_question_with_history(
    llm_choice: str,
    current_question: str,
    pair_history,
    history_window: int,
    temperature: float = 0.0,
) -> str:
    if not pair_history or int(history_window) <= 0:
        return current_question.strip()

    recent_history = format_recent_history(pair_history, int(history_window))

    system = (
        "You rewrite follow-up questions into standalone search queries for a RAG system. "
        "Do not answer the question. Only rewrite it. "
        "If the current question is already standalone, return it unchanged. "
        "Keep it concise and specific."
    )

    user = (
        f"RECENT CHAT HISTORY:\n{recent_history}\n\n"
        f"CURRENT QUESTION:\n{current_question}\n\n"
        "Standalone rewritten question:"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        rewritten, finish_reason = openai_style_call(
            llm_choice=llm_choice,
            messages=messages,
            temperature=temperature,
            max_tokens=256,
        )

        rewritten = rewritten.strip()

        if not rewritten or finish_reason == "error":
            return current_question.strip()

        return rewritten

    except Exception:
        return current_question.strip()


def validate_startup():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"MODEL_DIR not found: {MODEL_DIR}")

    if not os.path.exists(paths()["index"]) or not os.path.exists(paths()["meta"]):
        raise RuntimeError("KB not found. Run: python ingest_kb_3.py")


validate_startup()

print("[INFO] Loading embedding model, FAISS index, and metadata once...")
GLOBAL_EMBEDDER = SentenceTransformer(MODEL_DIR)
GLOBAL_INDEX = load_index()
GLOBAL_META_ROWS = load_meta()
print("[INFO] App resources loaded.")


def conversational_rag(
    message,
    pair_history,
    llm_choice,
    top_k,
    temperature,
    max_tokens,
    use_history_rewrite,
    history_window,
):
    if not message or not message.strip():
        return "Please enter a question.", []

    if llm_choice not in MODEL_CONFIGS:
        return f"Unknown model selection: {llm_choice}", []

    original_question = message.strip()

    if use_history_rewrite:
        retrieval_question = rewrite_question_with_history(
            llm_choice=llm_choice,
            current_question=original_question,
            pair_history=pair_history,
            history_window=int(history_window),
            temperature=0.0,
        )
    else:
        retrieval_question = original_question

    hits = retrieve(
        index=GLOBAL_INDEX,
        meta_rows=GLOBAL_META_ROWS,
        embedder=GLOBAL_EMBEDDER,
        query=retrieval_question,
        k=int(top_k),
    )

    context = build_context(hits)
    source_choices = build_source_choices(hits)

    if not context.strip():
        return "No relevant chunks retrieved. Try increasing top-k.", source_choices

    # answer = call_llm(
    #     llm_choice=llm_choice,
    #     question=retrieval_question,
    #     context=context,
    #     temperature=float(temperature),
    #     max_tokens=int(max_tokens),
    # )
    
    answer_question = (
    f"Original user question:\n{original_question}\n\n"
    f"Standalone retrieval question:\n{retrieval_question}\n\n"
    "Answer the original user question, using the retrieved context."
    )
    
    answer = call_llm(
        llm_choice=llm_choice,
        question=answer_question,
        context=context,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )

    final_answer = (
        f"{answer}\n\n"
        f"---\n"
        f"*Retrieval question used:* {retrieval_question}"
    )

    return final_answer, source_choices


def chat_submit(
    message,
    chat_history,
    llm_choice,
    top_k,
    temperature,
    max_tokens,
    use_history_rewrite,
    history_window,
):
    if chat_history is None:
        chat_history = []

    if not message or not message.strip():
        return chat_history, "", gr.update()

    pair_history = history_for_rewrite(chat_history)

    answer, source_choices = conversational_rag(
        message=message,
        pair_history=pair_history,
        llm_choice=llm_choice,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
        use_history_rewrite=use_history_rewrite,
        history_window=history_window,
    )

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, "", gr.update(choices=source_choices, value=None)


with gr.Blocks() as demo:
    gr.Markdown("# Conversational RAG App")

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=["Gemma", "Pixtral", "Qwen3-VL", "InternVL3.5"],
                value="Gemma",
                label="Model",
            )

            top_k = gr.Slider(
                1, 20,
                value=6,
                step=1,
                label="Top-k / K Neighbours",
                info="How many document chunks to retrieve from FAISS.",
            )

            history_window = gr.Slider(
                0, 10,
                value=2,
                step=1,
                label="Search Window / Chat History Window",
                info="How many previous conversation turns to use for rewriting follow-up questions. 0 = independent questions.",
            )

            temperature = gr.Slider(
                0.0, 1.0,
                value=0.2,
                step=0.05,
                label="Temperature",
                info="Lower = more factual, higher = more creative.",
            )

            max_tokens = gr.Slider(
                128, 2048,
                value=768,
                step=32,
                label="Max tokens",
                info="Maximum answer length.",
            )

            use_history_rewrite = gr.Checkbox(
                value=True,
                label="Use chat history for follow-up questions",
                info="Rewrites follow-up questions into standalone retrieval questions.",
            )

            gr.Markdown(
                "Ask follow-up questions naturally. The app rewrites follow-ups into standalone retrieval queries before FAISS search."
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=650,
            )

            chat_input = gr.Textbox(
                label="Question",
                placeholder="Ask a question about your ingested documents...",
                lines=2,
            )

            with gr.Row():
                submit_chat_btn = gr.Button("Submit", variant="primary")
                clear_chat_btn = gr.Button("Clear chat")

            source_selector = gr.Radio(
                label="Sources (click one to open the corresponding file location)",
                choices=[],
            )

    submit_chat_btn.click(
        fn=chat_submit,
        inputs=[
            chat_input,
            chatbot,
            model_choice,
            top_k,
            temperature,
            max_tokens,
            use_history_rewrite,
            history_window,
        ],
        outputs=[chatbot, chat_input, source_selector],
        show_progress="minimal",
    )

    chat_input.submit(
        fn=chat_submit,
        inputs=[
            chat_input,
            chatbot,
            model_choice,
            top_k,
            temperature,
            max_tokens,
            use_history_rewrite,
            history_window,
        ],
        outputs=[chatbot, chat_input, source_selector],
        show_progress="minimal",
    )

    source_selector.change(
        fn=open_selected_folder,
        inputs=[source_selector],
        outputs=[],
        show_progress="hidden",
    )

    clear_chat_btn.click(
        fn=lambda: ([], gr.update(choices=[], value=None)),
        inputs=[],
        outputs=[chatbot, source_selector],
        show_progress="hidden",
    )


demo.launch(server_name="127.0.0.1", share=False)