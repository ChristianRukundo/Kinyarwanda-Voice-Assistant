# app.py

import gradio as gr
from gtts import gTTS
import os
import tempfile
import time
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts.tts import gTTSError 


MODEL_ID = "benax-rw/KinyaWhisper" 
TTS_TEMP_DIR = tempfile.gettempdir() 
TTS_TEMP_PREFIX = "kinya_tts_" 
TARGET_ASR_SAMPLE_RATE = 16000 


print(f"Loading ASR model ({MODEL_ID}...)", flush=True) 
try:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval() 

    print("ASR model and processor loaded successfully.", flush=True)

except Exception as e:
    print(f"Error loading Hugging Face model ({MODEL_ID}): {e}", flush=True)
    print("Ensure transformers, torch, torchaudio are installed and model ID is correct.", flush=True)
    exit()

qa_pairs = {
    "muraho": "Muraho neza!",
    "witwa nde": "Ndi Gacurabwenge, umufasha wawe wa AI.",
    "rwanda coding academy iherereye he": "Iherereye mu Karere ka Nyabihu, mu Ntara y’Iburengerazuba.",
    "umurwa mukuru w'u rwanda ni uwuhe": "Umurwa mukuru w'u Rwanda ni Kigali.",
    "wavutse ryari": "Navutse vuba aha, ndi porogaramu ya orudinateri.",
    "ni ikihe kirunga kirekire mu rwanda": "Ikirunga kirekire kuruta ibindi mu Rwanda ni Kalisimbi.",
    "ikiyaga kinini mu rwanda ni ikihe": "Ikiyaga kinini mu Rwanda ni Ikiyaga cya Kivu.",
    "iteganyagihe ry'uyu munsi rimeze gute": "Iteganyagihe ry'uyu munsi rirerekana izuba ahantu henshi, ariko hashobora kubaho imvura nkeya ku gicamunsi.",
    "sa ngaho saa ndwi z'amanywa": "Ngaho saa ndwi z'amanywa.",
    "mfasha kugera i kigali": "Kugera i Kigali, ushobora gufata imodoka ziva aho uri cyangwa ugakoresha tagisi.",
    "urashobora kuvuga ikinyarwanda": "Yego, ndashobora kuvuga no kumva Kinyarwanda.",
    "waramutse": "Waramutse neza!",
    "wiriwe": "Wiriwe neza!",
    "amakuru": "Ni meza, wowe?",
    "igiciro cy'amazi ni angahe": "Igiciro cy'amazi kigenwa n'ikigo cy’igihugu gishinzwe ingufu.",
    "ni hehe nshobora kubona ibitabo by'ikinyarwanda": "Ushobora kubibona muri bibliotheque ya Kaminuza cyangwa ku rubuga rwa REB.",
    "urashobora kumfasha gukora research": "Yego, nshobora kugufasha gushakisha amakuru kuri internet cyangwa mu nyandiko zizewe.",
    "iki ni iki": "Mbwira neza icyo ushaka gusobanukirwa, ndagutega amatwi.",
    "umurongo wa internet urahari": "Yego, niba uri kuri murandasi, ushobora gukoresha application ukoresheje internet.",
    "ese covid yararangiye": "COVID-19 ntabwo yarangiye burundu, ariko yagenzuwe mu buryo bwinshi. Gukingirwa no kwirinda biracyakenewe.",
    "amakuru ya sport ni ayahe": "Amakuru ya sport y’uyu munsi arimo imikino ya shampiyona itandukanye, ushaka amakuru kuri kipe iyihe?",
    "amafaranga ya mobile money atangwa gute": "Ushobora kohereza amafaranga ukoresheje telefone, ukoresheje kode *182# kuri MTN cyangwa *150# kuri Airtel.",
    "ubuzima muri afurika buhagaze gute": "Ubuzima muri Afurika buragenda butera imbere ariko hari ibibazo by'ubukene n'ubuzima rusange bigikenewe gukemurwa.",
    "ese ejo hazaza h'ikoranabuhanga ni heza": "Yego, ikoranabuhanga rikomeje guteza imbere isi mu buryo bwihuse.",
    "ibyago byo gukoresha internet ni ibihe": "Hari ibyago byo kugirwaho ingaruka n’abatekamutwe, kwibwa amakuru, cyangwa kwanduzwa virusi.",
    "menya uko wacunga neza amafaranga": "Gira gahunda y’amafaranga ukoresha buri kwezi, wirinde imyenda, kandi ujye ubika ukoresheje banki cyangwa ikigo cy'imari.",
    "ubumenyi bwa siyansi bukora iki": "Bufasha gusobanukirwa iby’isi, guteza imbere ikoranabuhanga, n'ibindi byinshi bifitiye akamaro abantu.",
    "ni gute wabyara neza": "Gusura muganga, kwita ku mirire myiza, no gukurikiza inama z’ubuzima bifasha kubyara neza.",
    "ese abanyarwanda bose bavuga ikinyarwanda": "Yego, hafi ya bose bavuga Ikinyarwanda nk’ururimi rwabo kavukire.",
    "ni izihe ndimi zivugwa mu rwanda": "Ikinyarwanda, Igifaransa, Icyongereza n’Igiswahili.",
    "ikoranabuhanga ryafasha gute uburezi": "Rifasha kwigira kuri internet, gutanga amasomo hifashishijwe mudasobwa, no gusangira amakuru vuba.",
    "ndwaye umutwe": "Wakwivuza cyangwa kunywa imiti yabugenewe, ariko niba bikomeje wakagombye kujya kwa muganga.",
    "ese ikirere cyarahindutse": "Yego, hari impinduka zigaragara mu kirere, harimo ubushyuhe bwinshi n’imvura itunguranye, bitewe n’imihindagurikire y’ibihe.",
    "kompanyi ya mtn ikora iki": "MTN itanga serivisi za telefoni, internet, na Mobile Money mu Rwanda.",
    "ubushomeri mu rubyiruko ni iki": "Ni ikibazo cy’abantu batabona akazi, cyane cyane mu rubyiruko, nubwo hari gahunda zo kugabanya icyo kibazo.",
    "ntuye i muhanga": "Ni byiza! Muhanga ni umujyi ukura vuba kandi ufite ibikorwa byinshi by’ubucuruzi n’uburezi.",
    "ese imvura iragwa uyu munsi": "Rindira gato, reka ngerageze kugusubiza nkurikije iteganyagihe.",
    "amashanyarazi yabaye make": "Ibyo bishobora guterwa n’akazi k’ingufu, ibiza, cyangwa ikibazo cya tekiniki. Hamagara REG.",
    "umwaka mushya muhire": "Nawe umwaka mushya muhire, ibyiza byinshi bikubeho!",
    "igisirikare cy'u rwanda gikora iki": "Gishinzwe kurinda igihugu no gutabara aho gikenewe, harimo no gufasha abaturage mu bikorwa by’iterambere.",
    "ikibazo cya jenoside cyagize izihe ngaruka": "Cyasize ibikomere bikomeye mu muryango nyarwanda, ariko harakorwa byinshi mu kunga no kubaka igihugu bushya.",
    "mfite ikibazo cy'urukundo": "Mbwira ibikubangamiye, ndashobora kugufasha kuganira ku rukundo no kwiyakira.",
    "nkeneye akazi": "Shakisha ku mbuga z’amatangazo y’akazi cyangwa usure ibigo bitanga serivisi zo gushakira abantu akazi.",
    "intara y'amajyaruguru irimo uturere tune": "Yego, irimo Burera, Gakenke, Gicumbi, Musanze na Rulindo.",
    "cyane": "Ndumva ko ibyo bikunyuze cyane, ni byiza!",
    "ni ikihe gihugu kinini muri afurika": "Ni Algeria, gifite ubuso bunini kurusha ibindi.",
    "amashuri ya coding aboneka hehe": "Aboneka muri Rwanda Coding Academy, Andela, n’ahandi hatandukanye mu Rwanda.",
    "isi izarangira ryari": "Ntawe ubizi neza, ariko tugomba kubungabunga ibidukikije no kwita ku buzima bwacu.",
    "wumva amajwi yanjye": "Yego, ndakumva neza ukoresheje mikoro cyangwa dosiye y’amajwi.",
    "iki gitekerezo ni cyiza": "Urakoze! Komeza kugira ibitekerezo nk'ibi byubaka.",
    "urukundo ni iki": "Urukundo ni amarangamutima y’ubwuzu umuntu agirira undi, bigaragazwa n'ibikorwa no kwita ku wundi.",
    "ni iki cyakorwa ngo abantu babane neza": "Gufashanya, kuvugisha ukuri, kumva abandi no kubaha ni inkingi zo kubana neza.",
    "ndashaka kwiga ikoranabuhanga": "Ni byiza cyane! Tangirira ku masomo y'ibanze, ukoreshe internet cyangwa amashuri abihugura.",
    "ni iki gituma abana batiga": "Ibibazo birimo ubukene, imyumvire n'ibikorwa by'ingabo bishobora kubangamira uburezi.",
    "ese nshobora kubaza ikibazo icyo ari cyo cyose": "Yego rwose, mbaza icyo ushaka, ndahari ngo ngufashe.",
    "ese urashoboye gukora ibintu byinshi": "Yego, nshobora gusoma, kuvuga, gusubiza ibibazo no gutunganya amajwi n’inyandiko.",
    "mfashe umwanya wo kuganira": "Ndabishimira! Tuzagirana ibiganiro bifite umumaro.",
    "ni ryari amahugurwa ataha": "Ntabwo nzi igihe nyacyo, ariko wakurikirana amakuru ku mbuga zemewe z'ubuyobozi cyangwa imiryango y'amahugurwa.",
    "urashobora kunyigisha ikintu gishya": "Yego rwose! Mbwira icyo wifuza kwiga, tugitangire ako kanya.",
    "ese abana bakeneye kwitabwaho gute": "Bagomba guhabwa urukundo, kwitabwaho mu burezi, indyo iboneye n’uburere bwiza.",
    "ese ubuhinzi buteye imbere": "Yego, ariko haracyari ibibazo by'ibura ry'imbuto nziza n'isoko rihagije.",
    "ni izihe mpamvu twakoresha ikoranabuhanga mu bucuruzi": "Ryorohereza kugera ku isoko rinini, kugabanya ikiguzi, no gukorana n'abakiriya mu buryo bwihuse.",
    "ict bivuze iki": "ICT bisobanura Ikoranabuhanga mu Itumanaho n'Itangazamakuru.",
    "kompiyuteru ni iki": "Kompiyuteru ni igikoresho gifasha mu kubika, gutunganya no kohereza amakuru.",
    "internet ikora ite": "Internet ikora binyuze mu guhuza mudasobwa nyinshi ku isi, zishobora gusangira amakuru.",
    "software ni iki": "Software ni porogaramu igenga imikorere ya mudasobwa cyangwa igikoresho cy'ikoranabuhanga.",
    "hardware ni iki": "Hardware ni ibice bigaragara bya mudasobwa nk'ibikoresho by'inyuma n'imbere.","ni gute umuntu yirinda ibibazo by'ikoranabuhanga": "Ushobora kwirinda ibibazo by'ikoranabuhanga ukoresheje amagambo y'ibanga akomeye, gukoresha antivirus, no kwirinda gukanda ku butumwa butizewe.",
    "ni iki cyitwa phishing": "Phishing ni uburyo bwo kwiba amakuru y'umuntu binyuze mu butumwa bw'uburiganya bugamije kumushuka gutanga amakuru ye bwite.",
    "ni gute umuntu arinda amakuru ye kuri internet": "Ushobora kurinda amakuru yawe ukoresheje amagambo y'ibanga akomeye, kwirinda gusangiza amakuru y'ibanga, no gukoresha imiyoboro yizewe."
}
print(f"NLP dictionary loaded with {len(qa_pairs)} QA pairs.", flush=True)


def find_answer(question_text: str) -> str:
    """
    Finds an answer for the given question text using simple dictionary lookup.
    Performs basic normalization (lowercase, remove common punctuation).
    Returns a predefined answer or a default message if no exact match is found.
    """
    if not isinstance(question_text, str):
        print(f"Warning: Received non-string input for find_answer: {question_text}", flush=True)
        return "Mbabarira, habayeho ikibazo mu gusoma ikibazo cyawe."

    
    normalized_question = question_text.lower().strip().rstrip('?.!')
    print(f"Normalized question for matching: '{normalized_question}'", flush=True)

    
    answer = qa_pairs.get(normalized_question)

    if answer:
        print(f"Match found for key: '{normalized_question}'", flush=True)
        return answer
    else:
        print(f"No exact match found for normalized question: '{normalized_question}'", flush=True)
        return "Mbabarira, sinumvise neza ikibazo cyawe cyangwa sinzi igisubizo. Ushobora kugisubiramo mu bibazo bizwi?"


def transcribe_kinyarwanda(audio_filepath: str | None) -> str:
    """
    Transcribes Kinyarwanda audio using the loaded Hugging Face Kinyarwanda Whisper model.
    Returns the transcribed text or an error message.
    Requires ffmpeg and libsndfile for robust audio loading/processing.
    """
    if not audio_filepath:
        print("Error: No audio file path provided.", flush=True)
        return "Habayeho ikibazo: Nta dosiye y'amajwi yatanzwe."

    if not os.path.exists(audio_filepath):
        print(f"Error: Audio file not found at {audio_filepath}", flush=True)
        return f"Habayeho ikibazo: Dosiye y'amajwi ntiyabonetse: {os.path.basename(audio_filepath)}"

    print(f"Transcribing audio file: {audio_filepath}", flush=True)

    try:
        
        waveform, sample_rate = torchaudio.load(audio_filepath)

        
        if waveform.shape[0] > 1:
            print("Audio has multiple channels, converting to mono.", flush=True)
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        
        if sample_rate != TARGET_ASR_SAMPLE_RATE:
            print(f"Resampling audio from {sample_rate} Hz to {TARGET_ASR_SAMPLE_RATE} Hz", flush=True)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_ASR_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = TARGET_ASR_SAMPLE_RATE 

        
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")

        
        input_features = inputs.input_features.to(device)

        
        with torch.no_grad():
             
             predicted_ids = model.generate(input_features, max_length=256)

        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        print(f"Transcription result: '{transcription}'", flush=True)
        return transcription.strip()

    except Exception as e:
        print(f"Error during transcription: {e}", flush=True)
        
        error_detail = str(e).lower()
        if "libsndfile" in error_detail or "ffmpeg" in error_detail or "soundfile.LibsndfileError" in error_detail:
             return "Habaye ikibazo mu gusoma cyangwa gutunganya dosiye y'amajwi. Menya neza ko idosiye ari nzima kandi porogaramu zikenewe zifite (ffmpeg, libsndfile)."
        return f"Habaye ikibazo mu guhindura amajwi: {e}"


def speak_kinyarwanda(text_to_speak: str) -> str | None:
    """
    Converts Kinyarwanda text to speech using gTTS and saves it to a temporary MP3 file.
    Returns the path to the audio file or None if text is empty or generation fails.
    Handles gTTS language support errors specifically.
    """
    if not text_to_speak or not text_to_speak.strip():
        print("TTS received empty text, skipping.", flush=True)
        return None

    print(f"Generating speech for: '{text_to_speak}'", flush=True)

    tts_filepath = None
    try:
        tts = gTTS(text=text_to_speak, lang='rw', slow=False)
        
        fd, tts_filepath = tempfile.mkstemp(prefix=TTS_TEMP_PREFIX, suffix=".mp3", dir=TTS_TEMP_DIR)
        os.close(fd) 

        
        tts.save(tts_filepath)
        print(f"Speech saved to temporary file: {tts_filepath}", flush=True)
        return tts_filepath

    except gTTSError as e: 
        print(f"gTTS Error during generation: {e}", flush=True)
        if "Language not supported" in str(e):
            print("Kinyarwanda ('rw') is not supported by your gTTS installation. TTS will not work.", flush=True)
            
            return None
        
        raise
    except Exception as e:
        print(f"Unexpected Error during TTS generation: {e}", flush=True)
        
        if tts_filepath and os.path.exists(tts_filepath):
             try:
                 os.remove(tts_filepath)
                 print(f"Cleaned up temporary file: {tts_filepath}", flush=True)
             except OSError as cleanup_error:
                 print(f"Error cleaning up temp file {tts_filepath}: {cleanup_error}", flush=True)

        
        print("TTS failed entirely, cannot generate spoken error message.", flush=True)
        return None 


def cleanup_temp_files(directory: str = TTS_TEMP_DIR, prefix: str = TTS_TEMP_PREFIX):
    """Cleans up temporary audio files created by the TTS function."""
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".mp3"):
            filepath = os.path.join(directory, filename)
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Error removing temporary file {filepath}: {e}", flush=True)


def voice_assistant_pipeline(audio_input_path: str | None) -> tuple[str, str, str | None]:
    """
    Orchestrates the ASR -> NLP -> TTS pipeline.
    Takes an audio file path, returns transcription, answer text, and spoken answer audio path.
    Includes error handling and temporary file cleanup.
    """
    print("-" * 20, flush=True)
    start_time = time.time()

    cleanup_temp_files()

    transcribed_question = "Nta majwi yatanzwe cyangwa habayeho ikibazo." 
    answer_text = "Sinabonye igisubizo kubera ikibazo cyabaye." 
    spoken_answer_path = None

    try:

        transcribed_question = transcribe_kinyarwanda(audio_input_path)

        if "Habayeho ikibazo" in transcribed_question or "Nta majwi" in transcribed_question:
            print(f"ASR failed: {transcribed_question}", flush=True)
            answer_text = f"Ntibyakunze guhindura ijwi ryawe: {transcribed_question}"
            
            return transcribed_question, answer_text, None 

        asr_time = time.time()
        print(f"ASR Time: {asr_time - start_time:.2f}s", flush=True)

        
        answer_text = find_answer(transcribed_question)

        nlp_time = time.time()
        print(f"NLP Time: {nlp_time - asr_time:.2f}s", flush=True)

        spoken_answer_path = speak_kinyarwanda(answer_text)

        tts_time = time.time()
        print(f"TTS Time: {tts_time - nlp_time:.2f}s", flush=True)
        print(f"Total Time: {tts_time - start_time:.2f}s", flush=True)
        print("-" * 20, flush=True)


        return transcribed_question, answer_text, spoken_answer_path

    except Exception as e:
        print(f"An unexpected error occurred in the pipeline: {e}", flush=True)
        error_transcription = transcribed_question if "Habayeho ikibazo" not in transcribed_question else "Nta nyandiko kubera ikibazo cyabaye."
        error_answer = f"Habayeho ikibazo gikomeye cyane: {e}"
        return error_transcription, error_answer, None


print("Setting up Gradio interface...", flush=True)

custom_theme = gr.themes.Soft().set(
)


audio_input = gr.Audio(
    sources=["microphone", "upload"],
    type="filepath",
    label="1. Vuga ikibazo cyawe mu Kinyarwanda cyangwa wohereze dosiye (Speak your question in Kinyarwanda or upload a file)",
)
transcription_output = gr.Textbox(label="2. Icyo Navuze (Transcription from Kinyarwanda Whisper)", lines=2, interactive=False)
answer_output = gr.Textbox(label="3. Igisubizo Cyabonetse (Answer Found)", lines=2, interactive=False)
tts_output = gr.Audio(label="4. Igisubizo Kivuzwe (Spoken Answer)", type="filepath", autoplay=True, interactive=False)

example_files = [
    ["audio_samples/muraho.wav"],
    ["audio_samples/witwa_nde.wav"],
    ["audio_samples/rwanda_coding_academy.wav"],
    ["audio_samples/umurwa_mukuru.wav"],
    ["audio_samples/wavutse_ryari.wav"],
    ["audio_samples/kirunga_kirekire.wav"],
    ["audio_samples/ikiyaga_kinini.wav"],
]

iface = gr.Interface(
    fn=voice_assistant_pipeline,
    inputs=audio_input,
    outputs=[transcription_output, answer_output, tts_output],
    title="Mini Kinyarwanda Voice Assistant (Kinyarwanda Whisper)",
    description=(
        "Iki gikoresho gihindura ijwi ryawe mu Kinyarwanda rikaba inyandiko (hakoreshejwe 'benax-rw/Kinyarwanda Whisper'), "
        "gishaka igisubizo muri lisiti yatanzwe, hanyuma kikagusubiza mu ijwi.\n"
        "(This tool transcribes your Kinyarwanda speech using 'benax-rw/Kinyarwanda Whisper', "
        "finds a predefined answer, and speaks it back.)\n\n"
        "**Ibibazo Bizwi (Known Questions - Examples):**\n"
        f"Iyi porogaramu ishobora gusubiza ibibazo bizwi byabitswe (ubu harimo ibibazo {len(qa_pairs)}).\n"
        "Gerageza kuvuga kimwe muri ibi (cyangwa ukande kuri 'Examples'):\n"
        "- Muraho\n"
        "- Witwa nde?\n"
        "- Rwanda Coding Academy iherereye he?\n"
        "- Umurwa mukuru w'u Rwanda ni uwuhe?\n"
        "- Wavutse ryari?\n"
        "- Ni ikihe kirunga kirekire mu Rwanda?\n"
        "- Ikiyaga kinini mu Rwanda ni ikihe?"
        "\n\n**ICYITONDERWA (NOTE):** Gukoresha ijwi mu gusubiza birashobora kutagenda neza kuko ururimi rw'Ikinyarwanda rushobora kuba rudashyigikiwe na porogaramu ikoreshwa mu kuvuga inyandiko (gTTS). Ushobora gukoresha indi porogaramu nka Coqui TTS niba iboneka."
        "(**NOTE:** Spoken responses might not work because the Kinyarwanda language may not be supported by the text-to-speech program (gTTS). You might need to use another program like Coqui TTS if available.)"
    ),
    allow_flagging='never',
    examples=example_files if os.path.exists("audio_samples") and example_files else None,
    cache_examples=False, 
    theme=custom_theme,
)


print("Launching Gradio interface...", flush=True)
if __name__ == "__main__":
    iface.launch(share=True)
