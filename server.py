from flask import Flask, request, jsonify

from module.inference import TextClassifierInference
from module.tokenizer import Tokenizer

app = Flask(__name__)

ckpt_path = 'output/version_3/checkpoints/epoch=5-step=2928.ckpt'
vocab_dict_path = 'output/version_3/vocab_dict.json'

tokenizer = Tokenizer(pad_size=128)
tokenizer.load_vocab_dict(vocab_dict_path)

text_classifier = TextClassifierInference(
    ckpt_path=ckpt_path,
    tokenizer=tokenizer
)


@app.route('/classify', methods=['POST'])
def classify():
    """
    send post to /classify and return the classification results
    :return: {success: bool, message: str}. if 'success‘ is True, ‘message’ is the classification result
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"success": False, "message": "No text provided"})
    try:
        class_ = text_classifier.inference(data['text'], type_="label")
        return jsonify({"success": True, "message": class_})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
