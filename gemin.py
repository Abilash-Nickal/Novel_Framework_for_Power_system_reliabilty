from google import genai
import os

# 1. Setup: Initialize the Client
# The client automatically looks for the GEMINI_API_KEY environment variable.
try:
    client = genai.Client()
except Exception as e:
    # Handle the case where the API key might not be set or the client fails to initialize
    print(f"Error initializing the Gemini Client. Make sure your GEMINI_API_KEY environment variable is set.")
    print(f"Details: {e}")
    # Exit or return to avoid trying to call the API without a client
    exit()

# 2. Define the Model and Prompt
model_name = 'gemini-2.5-flash'
prompt = "Write a very short, cheerful story about a robot who discovers a sunflower."

# 3. Call the API to Generate Content
try:
    print(f"--- Sending Prompt to {model_name} ---")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    # 4. Display the Result
    print("\n--- Generated Story ---")
    print(response.text)
    print("-----------------------")

except Exception as e:
    print(f"An error occurred during the API call: {e}")