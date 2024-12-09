from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Initialize Hugging Face client
client = InferenceClient(api_key="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")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    messages = [
        {
            "role": "user",
            "content": user_message
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct",
            messages=messages,
            max_tokens=500
        )
        bot_response = completion.choices[0].message.get("content", "No response")
        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
