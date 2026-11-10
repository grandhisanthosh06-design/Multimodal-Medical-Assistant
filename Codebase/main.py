"""
main.py
--------
Multimodel Medical Assistant  (Capstone Project ID: HPPCS04)

WHAT THIS SCRIPT DOES
----------------------
For each of 5 input pairs (1 medical image + 1 text prescription/patient-details
file), this pipeline:
    1. Sends the medical image to a VISION-LANGUAGE MODEL, which returns a
       structured description of the visible clinical findings.
    2. Sends those findings together with the paired prescription/patient text
       to a TEXT-GENERATION LLM, which produces a realistic, multi-turn
       clinician-style conversation grounded in both the image and the text.
    3. Writes that conversation to a JSON file (one JSON file per input pair,
       5 files total).

TWO LLMs ARE USED (per assignment requirement), each responsible for a
different modality -- BOTH run fully locally via Ollama, so NO API KEY OR
INTERNET ACCESS IS REQUIRED at run time:
    - Vision-Language Model : "llava" served locally through Ollama, used to
      interpret the medical image. Chosen because it is a free, open-source
      vision-language model that runs entirely offline and can be queried
      through the same simple HTTP interface as the text model.
    - Text-Generation LLM   : "llama3" served locally through Ollama, used to
      turn the image findings + prescription text into a natural, multi-turn
      conversation. Chosen because it is free, runs fully offline, and is
      strong at fluent, instruction-following text generation.
If you would rather use different models (local or API-based), only the two
model-calling functions (`analyze_image_with_vision_model` and
`generate_conversation_with_text_model`) need to be edited -- the rest of the
pipeline (I/O, JSON schema, CLI, dashboard) does not need to change.

TWO WAYS TO RUN THIS SAME FILE
--------------------------------
This single file supports BOTH interfaces the assignment allows ("You can
use UI/command-line interface for your project as per your choice"):

    1. CLI / batch mode (the required entry point):
           python main.py --input_dir ./ --output_dir ./
       Processes all discovered pairs and writes conversation_<id>.json
       files to disk. This is what should be used for the actual graded
       batch run over the 5 submission pairs.

    2. Optional dashboard / chatbot UI mode:
           streamlit run main.py
       Opens a single-page browser dashboard ("Medical Assistant Pro") where
       you can pick a discovered patient case, run the pipeline on it, and
       view the generated conversation as styled chat bubbles -- or run the
       full batch and download each JSON from the browser.

The script automatically detects which mode it is running in (see the
`if __name__ == "__main__":` block at the bottom) -- no separate files, no
extra flags needed.

DIRECTORY / FILE CONVENTIONS
-----------------------------
Per the submission guidelines, everything lives in ONE directory (no
sub-directories) and is accessed using "./". Input files must be named so
that an image and its matching prescription/patient-details text share the
same numeric id, e.g.:

    patient_1.png   /  patient_1.jpg      <-> prescription_1.txt
    patient_2.png                          <-> prescription_2.txt
    ...
    patient_5.png                          <-> prescription_5.txt

Any image extension in {.png, .jpg, .jpeg} is accepted. The script
auto-discovers pairs by matching the trailing id number in the filename.

OUTPUT
------
For each pair with id "i", the script writes: conversation_i.json
    {
      "id": i,
      "image_file": "...",
      "prescription_file": "...",
      "image_findings": "<vision model output>",
      "conversation": [
        {"speaker": "Doctor", "text": "..."},
        {"speaker": "Patient", "text": "..."},
        ...
      ]
    }

DEPENDENCIES (install with pip before running)
------------------------------------------------
    pip install requests pillow
    pip install streamlit          # only needed for the optional dashboard

    Additionally, Ollama must be installed and running locally, with BOTH
    models pulled:
        ollama pull llava
        ollama pull llama3
        ollama serve      (usually starts automatically after install)

    No API key of any kind is required -- both models run on your machine.

USAGE
-----
    python main.py --input_dir ./ --output_dir ./        (CLI, required)
    streamlit run main.py                                 (optional dashboard)

    (see execution.txt for concrete examples of both)
"""

import os
import re
import json
import base64
import argparse
import sys

import requests


# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
VISION_MODEL_NAME = "llava"              # Local Ollama vision-capable model
TEXT_MODEL_NAME = "llama3"               # Local Ollama text-generation model
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
NUM_EXPECTED_PAIRS = 5


# ============================================================================
# CORE PIPELINE FUNCTIONS (used by both CLI mode and dashboard mode)
# ============================================================================

def find_input_pairs(input_dir):
    """
    Scan `input_dir` for image files and text files whose filenames end in
    the same numeric id (e.g. patient_1.png <-> prescription_1.txt), and
    return a sorted list of (id, image_path, text_path) tuples.

    Any image/text file whose name contains a trailing number is considered;
    files that cannot be matched to a partner are skipped with a warning.
    """
    images_by_id = {}
    texts_by_id = {}

    for fname in os.listdir(input_dir):
        full_path = os.path.join(input_dir, fname)
        if not os.path.isfile(full_path):
            continue

        name, ext = os.path.splitext(fname)
        match = re.search(r"(\d+)$", name)  # trailing digits = pair id
        if not match:
            continue
        pair_id = int(match.group(1))

        if ext.lower() in IMAGE_EXTENSIONS:
            images_by_id[pair_id] = full_path
        elif ext.lower() == ".txt":
            texts_by_id[pair_id] = full_path

    pairs = []
    for pair_id in sorted(set(images_by_id) & set(texts_by_id)):
        pairs.append((pair_id, images_by_id[pair_id], texts_by_id[pair_id]))

    missing = set(images_by_id) ^ set(texts_by_id)
    if missing:
        print(f"[WARN] Skipping unmatched ids (no image+text pair found): {sorted(missing)}")

    return pairs


def encode_image_to_base64(image_path):
    """
    Read an image file from disk and return its base64-encoded string, as
    required by Ollama's multimodal "images" field for vision-capable models.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image_with_vision_model(image_path):
    """
    LLM #1 (Vision-Language Model): send the medical image to the local
    "llava" model via Ollama and return a short, structured textual
    description of the clinically relevant visual findings (e.g. affected
    region, visible abnormality, general impression).

    Runs fully offline through Ollama -- no API key required. If Ollama /
    llava is not reachable, falls back to a clearly-labeled placeholder
    description so the rest of the pipeline can still be demonstrated
    end-to-end.
    """
    prompt = (
        "You are assisting with a clinical documentation pipeline. Look at this "
        "medical image and describe, in 3-5 concise sentences, the key visible "
        "findings a clinician would note (region imaged, any visible abnormality "
        "or notable feature, and overall impression). Do not give a definitive "
        "diagnosis -- describe only what is visually observable."
    )

    try:
        b64_image = encode_image_to_base64(image_path)
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": VISION_MODEL_NAME,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[WARN] Vision model (llava) call failed for {image_path} ({e}); using placeholder finding.")
        return ("[Vision model unavailable] Placeholder finding: an abnormality "
                "is visible in the imaged region; ensure Ollama is running to "
                "get real image analysis.")


def generate_conversation_with_text_model(image_findings, prescription_text, pair_id):
    """
    LLM #2 (Text-Generation Model): send the vision model's findings plus the
    patient's prescription/details text to a locally-hosted Llama 3 model
    (via Ollama's HTTP API) and ask it to produce a multi-turn clinician
    conversation grounded in both inputs. Returns a Python list of
    {"speaker": ..., "text": ...} turn dictionaries.

    If Ollama is not reachable, falls back to a minimal templated
    conversation so the pipeline still produces valid, well-formed output.
    """
    prompt = f"""You are generating a realistic clinical conversation for training data.

Image findings (from a vision model): {image_findings}

Patient prescription / details: {prescription_text}

Write a short, natural multi-turn conversation (6-10 turns) between a Doctor
and a Patient that is consistent with BOTH the image findings and the
prescription/details above. The doctor should reference the visual finding
and the patient's prescription context naturally.

Return ONLY valid JSON: a list of objects, each with keys "speaker" (either
"Doctor" or "Patient") and "text" (the line of dialogue). Do not include any
text outside the JSON list.
"""

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": TEXT_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()

        # The model may wrap the JSON in markdown fences; strip those if present.
        raw_text = re.sub(r"^```(json)?", "", raw_text.strip())
        raw_text = re.sub(r"```$", "", raw_text.strip())

        conversation = json.loads(raw_text)
        if isinstance(conversation, list):
            return conversation
        raise ValueError("Model did not return a JSON list.")

    except Exception as e:
        print(f"[WARN] Text model call failed for pair {pair_id} ({e}); using fallback conversation.")
        return [
            {"speaker": "Doctor", "text": f"Based on the imaging, I noted: {image_findings}"},
            {"speaker": "Patient", "text": "I see. What does that mean for my treatment?"},
            {"speaker": "Doctor", "text": f"Given your prescription details ({prescription_text[:120]}...), "
                                           f"we will proceed accordingly."},
            {"speaker": "Patient", "text": "Understood, thank you doctor."},
        ]


def build_output_record(pair_id, image_path, text_path, image_findings, conversation):
    """
    Assemble the final dictionary that will be serialized to
    conversation_<id>.json, combining metadata, the vision model's findings,
    and the generated conversation turns.
    """
    return {
        "id": pair_id,
        "image_file": os.path.basename(image_path),
        "prescription_file": os.path.basename(text_path),
        "image_findings": image_findings,
        "conversation": conversation,
    }


def save_json(record, output_dir, pair_id):
    """
    Write a single output record to <output_dir>/conversation_<pair_id>.json
    with indented, human-readable formatting.
    """
    out_path = os.path.join(output_dir, f"conversation_{pair_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {out_path}")


def process_pair(pair_id, image_path, text_path):
    """
    Run both models on a single (image, prescription) pair and return the
    finished output record. Shared by both CLI batch mode and the dashboard,
    so there is exactly one place the two-model pipeline is implemented.
    """
    with open(text_path, "r", encoding="utf-8") as f:
        prescription_text = f.read().strip()

    image_findings = analyze_image_with_vision_model(image_path)
    conversation = generate_conversation_with_text_model(image_findings, prescription_text, pair_id)
    return build_output_record(pair_id, image_path, text_path, image_findings, conversation)


# ============================================================================
# CLI / BATCH MODE  (required entry point: `python main.py ...`)
# ============================================================================

def run_cli():
    """
    Required CLI entry point: parse command-line arguments, discover input
    pairs, run both models on each pair, and write the JSON conversation
    output files. This is the mode graders/the batch submission will use.
    """
    parser = argparse.ArgumentParser(description="Multimodel Medical Assistant (HPPCS04)")
    parser.add_argument("--input_dir", type=str, default="./",
                         help="Directory containing the image + prescription .txt input files.")
    parser.add_argument("--output_dir", type=str, default="./",
                         help="Directory to write the conversation_<id>.json output files.")
    args = parser.parse_args()

    pairs = find_input_pairs(args.input_dir)
    if not pairs:
        print("[ERROR] No valid image/prescription pairs found in the input directory.")
        sys.exit(1)

    if len(pairs) != NUM_EXPECTED_PAIRS:
        print(f"[WARN] Expected {NUM_EXPECTED_PAIRS} pairs, found {len(pairs)}. Continuing anyway.")

    for pair_id, image_path, text_path in pairs:
        print(f"\n--- Processing pair {pair_id}: {image_path} + {text_path} ---")
        record = process_pair(pair_id, image_path, text_path)
        save_json(record, args.output_dir, pair_id)

    print("\nAll pairs processed.")


# ============================================================================
# OPTIONAL DASHBOARD / CHATBOT UI MODE  (`streamlit run main.py`)
# ============================================================================
# Everything in this section only runs if the script is launched through
# Streamlit (see the detection logic at the very bottom of the file). It is
# entirely optional -- the assignment allows either a UI or a CLI, and
# run_cli() above already satisfies the required "main.py that starts the
# code" entry point on its own.

def run_dashboard():
    """
    Optional dashboard entry point: a single-page Streamlit UI (no sidebar
    tab navigation) built on top of the same underlying pipeline used by
    run_cli(). Reuses process_pair() directly -- no pipeline logic is
    duplicated here, so the dashboard and the required CLI stay in sync.

    Everything lives on one page:
        - Select a discovered patient case (image + prescription pair)
        - View the image and its referring notes
        - Run the pipeline on that case and see the image findings +
          generated Doctor/Patient conversation as styled chat bubbles
        - Upload a brand-new X-ray/image + prescription (file or typed
          text) to analyze a case that isn't part of the 5 discovered pairs
        - Download that case's JSON, or run the full batch of all
          discovered cases in one action
    """
    import streamlit as st

    st.set_page_config(page_title="Medical Assistant Pro", page_icon="\U0001FA7A", layout="wide")

    # Custom CSS for the chat-bubble / clinical-card styling.
    st.markdown("""
        <style>
        .main { background-color: #f9f9ff; }
        .stButton>button { width: 100%; border-radius: 8px; background-color: #0066cc; color: white; }
        .clinical-card { background-color: white; padding: 20px; border-radius: 12px; border-left: 4px solid #0066cc; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; }
        .chat-bubble-doctor { background-color: #eef2ff; padding: 12px; border-radius: 12px 12px 12px 0; margin-bottom: 10px; border: 1px solid #cfdaf2; }
        .chat-bubble-patient { background-color: white; padding: 12px; border-radius: 12px 12px 0 12px; margin-bottom: 10px; border: 1px solid #e2e8f0; text-align: right; }
        </style>
    """, unsafe_allow_html=True)

    st.title("\U0001FA7A Medical Assistant Pro")

    input_dir = "./"
    pairs = find_input_pairs(input_dir)
    completed_ids = st.session_state.setdefault("completed_ids", set())

    if not pairs:
        st.warning("No image/prescription pairs found in the current directory.")
        return

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("Patient Record")
        selected_idx = st.selectbox(
            "Select Patient Case", range(len(pairs)),
            format_func=lambda i: f"Case ID: PT-847{pairs[i][0]}",
        )
        pid, img_path, txt_path = pairs[selected_idx]

        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img_path, use_container_width=True)
        with c2:
            st.markdown(f"**ID: PT-847{pid}**")
            status = "Analyzed" if pid in completed_ids else "Not yet analyzed"
            st.caption(status)
            with open(txt_path, "r", encoding="utf-8") as f:
                st.caption("Referring Notes")
                st.text(f.read()[:200])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if st.button("Run Analysis Pipeline"):
            with st.spinner("Analyzing image and generating dialogue..."):
                record = process_pair(pid, img_path, txt_path)
            completed_ids.add(pid)
            st.session_state[f"record_{pid}"] = record

        record = st.session_state.get(f"record_{pid}")
        if record:
            st.subheader("Image Findings")
            st.markdown(
                f'<div class="clinical-card" style="border-left-color: #10b981;">{record["image_findings"]}</div>',
                unsafe_allow_html=True,
            )

            st.subheader("Patient Dialogue")
            for turn in record["conversation"]:
                role_class = "chat-bubble-doctor" if turn.get("speaker") == "Doctor" else "chat-bubble-patient"
                st.markdown(
                    f'<div class="{role_class}"><b>{turn.get("speaker", "Unknown")}:</b> {turn.get("text", "")}</div>',
                    unsafe_allow_html=True,
                )

            st.download_button(
                "Download conversation JSON",
                data=json.dumps(record, indent=2),
                file_name=f"conversation_{pid}.json",
                mime="application/json",
            )
        else:
            st.info("Click 'Run Analysis Pipeline' to view diagnostic results for this case.")

    st.markdown("---")
    st.subheader("Upload New Case")
    st.write("Upload a new X-ray/medical image and its prescription/patient details to generate a "
             "conversation on the spot, without needing to add files to the input directory first.")

    upload_col1, upload_col2 = st.columns([1, 1.2])

    with upload_col1:
        uploaded_image = st.file_uploader("Medical image (X-ray, MRI, etc.)",
                                           type=["png", "jpg", "jpeg"], key="upload_image")
        uploaded_prescription_file = st.file_uploader("Prescription / patient details (.txt) -- optional",
                                                        type=["txt"], key="upload_prescription_file")
        manual_prescription_text = st.text_area(
            "Or type the prescription / patient details here",
            height=140, key="upload_manual_text",
            placeholder="e.g. 40-year-old male, complains of persistent cough for a week...",
        )

    with upload_col2:
        if st.button("Analyze Uploaded Case"):
            if uploaded_image is None:
                st.error("Please upload a medical image first.")
            else:
                # Prefer an uploaded prescription .txt file's contents; fall back to the typed text.
                if uploaded_prescription_file is not None:
                    prescription_text = uploaded_prescription_file.read().decode("utf-8", errors="ignore").strip()
                    prescription_display_name = uploaded_prescription_file.name
                else:
                    prescription_text = manual_prescription_text.strip()
                    prescription_display_name = "manual_entry.txt"

                if not prescription_text:
                    st.error("Please upload a prescription .txt file or type the patient details.")
                else:
                    # Save the uploaded image to a temporary flat file so the existing
                    # path-based pipeline functions can read it, same as any other input.
                    temp_image_path = f"_uploaded_{uploaded_image.name}"
                    with open(temp_image_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())

                    st.image(temp_image_path, caption="Uploaded image", width=300)

                    with st.spinner("Analyzing image and generating dialogue..."):
                        image_findings = analyze_image_with_vision_model(temp_image_path)
                        conversation = generate_conversation_with_text_model(
                            image_findings, prescription_text, "uploaded"
                        )
                        upload_record = build_output_record(
                            "uploaded", temp_image_path, prescription_display_name,
                            image_findings, conversation,
                        )

                    os.remove(temp_image_path)  # clean up the temporary upload

                    st.subheader("Image Findings")
                    st.markdown(
                        f'<div class="clinical-card" style="border-left-color: #10b981;">{image_findings}</div>',
                        unsafe_allow_html=True,
                    )

                    st.subheader("Patient Dialogue")
                    for turn in conversation:
                        role_class = "chat-bubble-doctor" if turn.get("speaker") == "Doctor" else "chat-bubble-patient"
                        st.markdown(
                            f'<div class="{role_class}"><b>{turn.get("speaker", "Unknown")}:</b> {turn.get("text", "")}</div>',
                            unsafe_allow_html=True,
                        )

                    st.download_button(
                        "Download conversation JSON",
                        data=json.dumps(upload_record, indent=2),
                        file_name="conversation_uploaded.json",
                        mime="application/json",
                    )

    st.markdown("---")
    if st.button("Run Full Batch (all discovered cases)"):
        for p_id, i_path, t_path in pairs:
            st.markdown(f"### PT-847{p_id}")
            with st.spinner(f"Processing PT-847{p_id}..."):
                rec = process_pair(p_id, i_path, t_path)
            completed_ids.add(p_id)
            st.session_state[f"record_{p_id}"] = rec
            for turn in rec["conversation"]:
                role_class = "chat-bubble-doctor" if turn.get("speaker") == "Doctor" else "chat-bubble-patient"
                st.markdown(
                    f'<div class="{role_class}"><b>{turn.get("speaker", "Unknown")}:</b> {turn.get("text", "")}</div>',
                    unsafe_allow_html=True,
                )
            st.download_button(
                f"Download conversation_{p_id}.json",
                data=json.dumps(rec, indent=2),
                file_name=f"conversation_{p_id}.json",
                mime="application/json",
                key=f"batch_dl_{p_id}",
            )
            st.divider()


# ============================================================================
# ENTRY POINT: automatically choose CLI mode vs dashboard mode
# ============================================================================
if __name__ == "__main__":
    # `streamlit run main.py` executes this file as __main__ just like a
    # normal `python main.py` call would, so we distinguish the two by
    # checking whether a live Streamlit runtime is actually driving this
    # script. If it is, show the dashboard; otherwise fall back to the
    # required CLI/batch pipeline.
    _running_under_streamlit = False
    try:
        import streamlit as _st
        _running_under_streamlit = _st.runtime.exists()
    except Exception:
        _running_under_streamlit = False

    if _running_under_streamlit:
        run_dashboard()
    else:
        run_cli()