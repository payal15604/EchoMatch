from Speech2Text import Speech_to_Text
from Text2Text import compare_texts

text = input('Enter Audio Path')
audio_path = input('Enter Audio Path')
def main(audio_path, text):
    audio_path = audio_path
    text = text
    answer_text = Speech_to_Text(path_string)
    similarity_score = compare_texts(text, answer_text)
    print("=====================================================")
    print('The percentage score for correct pronounciation is ', similarity_score)
    print("=====================================================")
    
