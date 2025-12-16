from nodetool.nodes.gemini.audio import TextToSpeech, VoiceName

def test_voice_name_validation():
    # Test valid voice
    node = TextToSpeech(text="hello", voice_name="puck")
    assert node.voice_name == VoiceName.PUCK
    
    # Test valid voice case insensitive
    node = TextToSpeech(text="hello", voice_name="PUCK")
    assert node.voice_name == VoiceName.PUCK
    
    # Test invalid voice defaults to KORE
    node = TextToSpeech(text="hello", voice_name="invalid_voice_name")
    assert node.voice_name == VoiceName.KORE
    
    # Test "Nova" (which was the cause of the error)
    node = TextToSpeech(text="hello", voice_name="Nova")
    assert node.voice_name == VoiceName.KORE
    
    print("All validation tests passed!")

if __name__ == "__main__":
    test_voice_name_validation()
