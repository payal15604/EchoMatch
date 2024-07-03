from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
import librosa

# Function to transcribe audio
def Speech_to_Text(audio_path):
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    if model.training:
        print("Training the model...")
    
    else:
        print("Model is ready for inference.")
    
    
    speech, sample_rate = sf.read(audio_path)

    # Resample the audio to 16000 Hz if necessary
    if sample_rate != 16000:
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Ensure the audio is in the correct format
    input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values

    # Perform inference with the model
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

#model_name = "facebook/wav2vec2-base-960h"
#processor = Wav2Vec2Processor.from_pretrained(model_name)
#model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Test the transcription function
#audio_file_path = "positive_result.mp3"
#transcription = Speech_To_Text(audio_file_path)
#print("Transcribed text:", transcription)
