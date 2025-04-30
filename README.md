# 🇷🇼 Mini Kinyarwanda Voice Assistant (using KinyaWhisper) 🤖️🎙️

Murakaza neza! Welcome to the Mini Kinyarwanda Voice Assistant project. This application demonstrates a simple voice interaction system for the Kinyarwanda language, simulating how a robot might hear, understand, and respond.

This project fulfills the assignment requirements by:

- 👂 **Hearing:** Using the `benax-rw/KinyaWhisper` model for Kinyarwanda Speech-to-Text (ASR).
- 🧠 **Understanding:** Matching the transcribed text to predefined questions using basic dictionary lookup (NLP).
- 🗣️ **Speaking:** Generating spoken Kinyarwanda answers using Google Text-to-Speech (TTS).

## 🎯 Project Goal

To build a functional prototype showcasing core voice AI components (ASR, NLP, TTS) for Kinyarwanda, suitable for demonstrating basic voice interaction in applications like robotics.

## 📁 Folder Structure

This project follows standard conventions, placing source code in `src/`:

.
├── audio_samples/ # 🎧 Your Kinyarwanda audio recordings (.wav/.mp3)
│ ├── muraho.wav
│ ├── witwa_nde.wav
│ └── ... (at least 5 samples)
├── src/ # 🐍 Source code directory
│ └── app.py # Main application script
├── .gitignore # (Optional) Files to ignore for Git
├── LICENSE # (Optional) License file
├── README.md # 📄 This explanation file
├── requirements.txt # 📦 Python package dependencies
└── venv/ # (Optional) Python virtual environment folder

## ✨ How It Works: Code Breakdown

The `src/app.py` script orchestrates the entire process:

1.  **Initialization & Model Loading:**

    - Imports necessary libraries (`gradio`, `gtts`, `torch`, `torchaudio`, `transformers`).
    - Defines constants like the Hugging Face `MODEL_ID` (`benax-rw/KinyaWhisper`).
    - Loads the `WhisperProcessor` and `WhisperForConditionalGeneration` model from Hugging Face, automatically downloading them if needed. It detects if a GPU (`cuda`) is available for faster processing.
      ```python
      # Loads processor and model
      processor = WhisperProcessor.from_pretrained(MODEL_ID)
      model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
      model.eval() # Sets model to evaluation mode
      ```

2.  **👂 Speech Recognition (ASR - `transcribe_kinyarwanda` function):**

    - Takes the audio file path as input.
    - Loads the audio using `torchaudio.load()`.
    - Converts stereo audio to mono.
    - Resamples the audio to the required `TARGET_ASR_SAMPLE_RATE` (16000 Hz) if necessary.
    - Uses the `processor` to prepare the audio features for the model.
    - Feeds the features into the `model.generate()` method to get predicted token IDs.
    - Decodes the token IDs back into Kinyarwanda text using `processor.batch_decode()`.
      ```python
      # Inside transcribe_kinyarwanda:
      waveform, sample_rate = torchaudio.load(audio_filepath)
      # ... (resampling/mono conversion) ...
      inputs = processor(waveform.squeeze().numpy(), ...)
      input_features = inputs.input_features.to(device)
      predicted_ids = model.generate(input_features, ...)
      transcription = processor.batch_decode(predicted_ids, ...)[0]
      ```

3.  **🧠 Natural Language Processing (NLP - `find_answer` function):**

    - Takes the transcribed text from the ASR step.
    - Normalizes the text (lowercase, remove trailing punctuation, strip whitespace).
    - Looks for an exact match of the normalized text in the keys of the predefined `qa_pairs` dictionary.
      ```python
      # Inside find_answer:
      normalized_question = question_text.lower().strip().rstrip('?.!')
      answer = qa_pairs.get(normalized_question) # Efficient dictionary lookup
      ```
    - Returns the corresponding answer if found, otherwise returns a default "I don't understand" message.

4.  **🗣️ Text-to-Speech (TTS - `speak_kinyarwanda` function):**

    - Takes the answer text from the NLP step.
    - Uses the `gTTS` library to convert the text into speech (`lang='rw'`).
    - Saves the generated speech as a temporary `.mp3` file.
    - Returns the path to the temporary audio file.
    - Includes error handling, especially for potential `gTTS` language support issues.
      ```python
      # Inside speak_kinyarwanda:
      tts = gTTS(text=text_to_speak, lang='rw', slow=False)
      # ... (create temporary file) ...
      tts.save(tts_filepath)
      ```

5.  **🌐 Web Interface (Gradio - `iface` object):**

    - Creates an interactive UI using `gradio.Interface`.
    - Defines input components (`gr.Audio` for microphone/upload).
    - Defines output components (`gr.Textbox` for transcription/answer, `gr.Audio` for spoken response).
    - Connects the UI elements to the main `voice_assistant_pipeline` function.
    - Includes a title, description, examples (using your `audio_samples`), and a theme.
    - Adds a crucial note about potential `gTTS` limitations for Kinyarwanda TTS output.

6.  **🧹 Cleanup (`cleanup_temp_files` function):**
    - A helper function to remove temporary TTS audio files created during previous runs, keeping the temp directory clean. Called at the start of the main pipeline.

## 🚀 Setup and Installation

Follow these steps to get the assistant running:

1.  **Clone Your Repository:**

    ```bash
    # Clone your specific repository
    git clone [https://github.com/ChristianRukundo/Kinyarwanda-Voice-Assistant.git](https://github.com/ChristianRukundo/Kinyarwanda-Voice-Assistant.git)
    cd Kinyarwanda-Voice-Assistant
    ```

2.  **🛠️ Install System Tools:**

    - **Python 3:** Make sure you have Python 3.8+ installed. Check with `python3 --version`.
    - **ffmpeg:** Needed for audio processing by `torchaudio`.
      - 🐧 Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
      - 🍎 macOS (Homebrew): `brew install ffmpeg`
      - 🪟 Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` folder to your system's PATH.
    - **libsndfile (Recommended):** Helps read `.wav` files reliably.
      - 🐧 Ubuntu/Debian: `sudo apt install libsndfile1`
      - 🍎 macOS (Homebrew): `brew install libsndfile`
      - 🪟 Windows: Often included with Anaconda or some Python wheels. Install if you face issues reading WAV files.

3.  **🐍 Create Python Virtual Environment:** (Keeps dependencies tidy!)

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

4.  **📦 Install Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```
    _⏳ This might take a while as it downloads libraries like `torch`, `transformers`, and the KinyaWhisper model files on the first run._

## ▶️ How to Run

1.  **🎙️ Record Your Audio:** Ensure the `audio_samples/` folder contains at least 5 audio files (`.wav` or `.mp3`) speaking the _exact_ Kinyarwanda questions listed as keys in the `qa_pairs` dictionary in `src/app.py`. _(Your repo already has these! 👍)_
2.  **Activate Environment:** If not already active:
    ```bash
    source venv/bin/activate # Linux/macOS or venv\Scripts\activate for Windows
    ```
3.  **🚀 Launch the App:** (Note: Run the script from the `src` directory)
    ```bash
    python app.py
    ```
    _🌐 Wait for the Gradio interface link (like `http://127.0.0.1:7860`) to appear in your terminal._
4.  **🗣️ Interact:** Open the link in your browser. Use the microphone or upload an audio file from `audio_samples/`. See the transcription, answer, and hear the spoken response!


## 🙏 Model Acknowledgment

This project proudly uses the `benax-rw/KinyaWhisper` model, fine-tuned by Gabriel Baziramwabo. Thank you for providing this valuable resource for the Kinyarwanda language!

- **Model Link:** [https://huggingface.co/benax-rw/KinyaWhisper](https://huggingface.co/benax-rw/KinyaWhisper)

---

