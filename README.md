🩺 Multimodal Medical Assistant (HPPCS04)An end-to-end Multimodal Clinical Documentation & Dialogue Generation Pipeline that pairs medical imaging (MRI/radiographs) with referring clinical notes and prescriptions to generate structured visual findings and realistic, multi-turn Doctor–Patient consultations.Built to run 100% locally and offline using open-source Vision-Language and Text-Generation Large Language Models served via Ollama—requiring zero paid API keys or external cloud dependencies.✨ Key FeaturesDual-LLM Multimodal Architecture:Vision-Language Model (llava): Analyzes medical scans (patient_<id>.png/.jpg) and extracts concise, clinically grounded visual observations.Text-Generation Model (llama3): Synthesizes the visual findings together with the patient's referring notes (prescription_<id>.txt) into a natural 6–10 turn Doctor–Patient dialogue.Dual Execution Modes in a Single File (main.py):CLI / Batch Mode: Automatically discovers all matched (image, prescription) pairs in the directory and generates structured conversation_<id>.json files.Interactive Streamlit Dashboard ("Medical Assistant Pro"): A single-page web UI to inspect patient cases, view styled Doctor/Patient chat bubbles, upload custom medical scans on the fly, and export JSON records.Fault-Tolerant Pipeline: Includes graceful fallbacks if local model servers are unreachable, guaranteeing valid JSON schema output for automated grading and testing.🏗️ System Architecture+-----------------------+       +-------------------------+
|   Medical Scan Image  |       |  Prescription / Notes   |
|    (patient_i.png)    |       |   (prescription_i.txt)  |
+-----------+-----------+       +------------+------------+
            |                                |
            v                                |
+-----------------------+                    |
|  Vision LLM (LLaVA)   |                    |
|   via Local Ollama    |                    |
+-----------+-----------+                    |
            |                                |
            |  Structured Image Findings     |
            +---------------+----------------+
                            |
                            v
               +-------------------------+
               |   Text LLM (Llama 3)    |
               |    via Local Ollama     |
               +------------+------------+
                            |
                            v
               +-------------------------+
               |  Multi-Turn Clinical    |
               |  Dialogue JSON Output   |
               |  (conversation_i.json)  |
               +-------------------------+
📂 Project StructureAll project files reside in a flat root directory per capstone specification:.
├── main.py                  # Core multimodal pipeline (CLI batch runner + Streamlit UI)
├── execution.txt            # Quick-start execution instructions
├── README.md                # Project documentation
├── patient_1.png            # Input medical scan (Case 1)
├── patient_2.png            # Input medical scan (Case 2)
├── patient_3.png            # Input medical scan (Case 3)
├── patient_4.png            # Input medical scan (Case 4)
├── patient_5.png            # Input medical scan (Case 5)
├── prescription_1.txt       # Referring clinical notes & prescription (Case 1)
├── ...                      # prescriptions 2 through 5
├── conversation_1.json      # Generated findings & Doctor-Patient dialogue (Case 1)
├── conversation_2.json      # Generated findings & Doctor-Patient dialogue (Case 2)
├── conversation_3.json      # Generated findings & Doctor-Patient dialogue (Case 3)
├── conversation_4.json      # Generated findings & Doctor-Patient dialogue (Case 4)
└── conversation_5.json      # Generated findings & Doctor-Patient dialogue (Case 5)
⚙️ Prerequisites & Installation1. Install and Start OllamaDownload and install Ollama, then pull the required vision and text models in your terminal:ollama pull llava
ollama pull llama3
(Ensure the Ollama server is running locally at http://localhost:11434, which starts automatically after installation or by running ollama serve).2. Install Python DependenciesMake sure you have Python 3.8+ installed, then install the required packages:pip install requests pillow streamlit
🚀 Usagemain.py automatically detects whether it is invoked from the command line or through Streamlit.Mode 1: CLI / Batch Mode (Standard Execution)To process all 5 patient image/prescription pairs in bulk and write conversation_1.json through conversation_5.json:python main.py --input_dir ./ --output_dir ./
Mode 2: Interactive Web Dashboard ("Medical Assistant Pro")To launch the visual clinical dashboard in your browser:streamlit run main.py
From the dashboard, you can:Browse Discovered Cases: Select any patient ID (PT-8471 – PT-8475) to view their scan and referring notes side-by-side.Run Analysis Pipeline: Generate visual findings and view the Doctor–Patient dialogue rendered in styled chat bubbles.Upload Custom Cases: Upload any new X-ray/MRI image (.png, .jpg, .jpeg) and type or upload custom patient notes for instant analysis.Export JSON: Download individual or full-batch conversation_<id>.json files directly from the browser.📄 Output JSON SchemaEach processed case produces a conversation_<id>.json file structured as follows:{
  "id": 4,
  "image_file": "patient_4.png",
  "prescription_file": "prescription_4.txt",
  "image_findings": "The image appears to be an MRI brain scan with a focus on the frontal region...",
  "conversation": [
    {
      "speaker": "Doctor",
      "text": "So, let's take a look at your MRI scan. What do you see here?"
    },
    {
      "speaker": "Patient",
      "text": "Uh, it looks like my brain... it's all lopsided, and there's this weird growth thingy on the right side?"
    }
  ]
}
🛠️ ConfigurationYou can customize model names, endpoints, or supported file extensions at the top of main.py:ConstantDefault ValueDescriptionVISION_MODEL_NAME"llava"Local Ollama vision-language model for scan analysisTEXT_MODEL_NAME"llama3"Local Ollama LLM for clinical dialogue synthesisOLLAMA_GENERATE_URL"http://localhost:11434/api/generate"Local REST endpoint for Ollama inferenceIMAGE_EXTENSIONS(".png", ".jpg", ".jpeg")Supported medical image formatsNUM_EXPECTED_PAIRS5Expected number of matched input cases
