from langdetect import detect, DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator


def detect_and_transform(sentence:str) ->dict:
    try:
        lang_id = detect(sentence)
        
        if lang_id != "en":
             translated = GoogleTranslator(source='auto', target='en').translate(sentence)
             return {
                 "original_text":sentence,
                 "translated_text":translated,
                 "original_text_id":"en"
             }
        else:
            return {
                "original_text":sentence
            }
    except Exception as e:
        return {
            "The error is Comming From MS section":str(e)
        }
    except LangDetectException:
        return {
            "Cannot find out and Language ID"
        }
        
def transform_to_origin(sentence: str, target_lang_id: str) -> dict:
    try:
        lang_id = detect(sentence)
        
        if lang_id == "en":
            translated = GoogleTranslator(source='auto', target=lang_id).translate(sentence)
            return {
                "translated_text":translated
            }
            
    except Exception as e:
        return {
            "The error is Comming From MS section":str(e)
        }
    except LangDetectException:
        return {
            "Cannot find out and Language ID"
        }
            