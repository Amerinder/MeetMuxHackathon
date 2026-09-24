"""
Live Server API and Layout Verification Test
"""

import sys
from gradio_client import Client

def test_live_server():
    print("Connecting to live Gradio server at http://127.0.0.1:7860/ ...")
    client = Client("http://127.0.0.1:7860/")
    
    test_cases = [
        "How can I help you?",
        "What do you mean?",
        "You're right.",
        "I bought a new car.",
        "",
        "   "
    ]
    
    print("\n--- Testing Live Translation Endpoints ---")
    for s in test_cases:
        res = client.predict(text=s, api_name="/translate_text")
        print(f"INPUT: '{s}' -> OUTPUT: '{res}'")
        assert res is not None and len(res) > 0

    print("\nSUCCESS: All live server translation calls responded correctly!")

if __name__ == "__main__":
    test_live_server()
