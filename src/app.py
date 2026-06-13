"""
app.py
------
Flask web server for MedBot.

Routes:
    GET  /            -> serves the chat web page (templates/index.html)
    POST /upload      -> receives a PDF, builds the knowledge base from it
    POST /chat        -> answers a question using the current knowledge base
    GET  /status      -> reports whether a document is currently loaded

The heavy models are loaded ONCE when the server starts (see the global
`bot`). Uploading a new PDF only rebuilds the lightweight FAISS index, so it
is fast and does not reload the models.

Usage:
    python src/app.py
    then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import tempfile

from flask import Flask, request, jsonify, render_template

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import MedBot

# ---------------------------------------------------------------------------
# Flask app + one shared MedBot instance (models loaded once at startup)
# ---------------------------------------------------------------------------
# app.py lives in src/, but templates/ sits at the project root, so we point
# Flask's template folder there explicitly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Limit uploads to 32 MB to avoid memory problems with huge PDFs.
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

print("Starting MedBot web server - loading models, please wait...")
# load_existing=False: start with an empty knowledge base; the user supplies
# a PDF through the web page. (Set to True to also pick up any pre-built index.)
bot = MedBot(load_existing=False)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Serve the single-page chat interface."""
    return render_template("index.html")


@app.route("/status")
def status():
    """Tell the front-end whether a document is loaded (used on page load)."""
    return jsonify({
        "ready": bot.has_knowledge(),
        "source": bot.source_name,
    })


@app.route("/upload", methods=["POST"])
def upload():
    """
    Receive a PDF file, build the knowledge base from it, and report back.
    The file is processed from a temporary path and not kept on disk.
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file was uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False,
                        "error": "Only PDF files are supported."}), 400

    # Save to a temporary file, index it, then remove it.
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        file.save(tmp_path)

        n_chunks = bot.index_pdf(tmp_path)
        # index_pdf stores only the basename; show the real uploaded name.
        bot.source_name = file.filename

        return jsonify({
            "ok": True,
            "source": file.filename,
            "chunks": n_chunks,
            "message": f"'{file.filename}' is ready. Indexed {n_chunks} "
                       f"sections. Ask me anything about it.",
        })
    except ValueError as err:
        # Raised by index_pdf for empty / scanned PDFs
        return jsonify({"ok": False, "error": str(err)}), 422
    except Exception as err:  # noqa: BLE001 - surface any other failure cleanly
        return jsonify({"ok": False,
                        "error": f"Could not process the PDF: {err}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/chat", methods=["POST"])
def chat():
    """Answer a single question with the RAG engine."""
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"answer": "Please type a question."})

    answer = bot.answer(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    # debug=False so the models are not loaded twice by the reloader.
    app.run(host="127.0.0.1", port=5001, debug=False)
