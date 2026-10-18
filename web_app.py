"""Local web interface for the research-only clinical RAG prototype."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from MedicalAgent import DISCLAIMER, build_assistant


PROJECT_DIR = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_DIR / "static"
MAX_REQUEST_BYTES = 16_384
LOGGER = logging.getLogger("clinical_rag.web")


def _serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    triage = result["triage"]
    interaction = result["interaction_assessment"]

    return {
        "answer": result["answer"],
        "review_required": result["review_required"],
        "triage": triage.model_dump(mode="json"),
        "interaction": interaction.model_dump(mode="json"),
        "sources": [
            {
                "id": document.metadata["source_id"],
                "title": document.metadata["source"],
                "url": document.metadata["dataset_url"],
                "excerpt": document.page_content,
            }
            for document in result.get("evidence", [])
        ],
        "drug_labels": [
            {
                "id": label["source_id"],
                "medication": label["medication"],
                "status": label["status"],
                "url": label["url"],
            }
            for label in result.get("drug_labels", [])
        ],
        "safety_notes": result.get("safety_notes", []),
        "disclaimer": DISCLAIMER,
    }


def create_handler(assistant):
    class ClinicalChatHandler(BaseHTTPRequestHandler):
        server_version = "ClinicalRAGDemo/1.0"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            static_files = {
                "/": STATIC_DIR / "index.html",
                "/static/style.css": STATIC_DIR / "style.css",
                "/static/app.js": STATIC_DIR / "app.js",
            }
            file_path = static_files.get(self.path)
            if file_path is None:
                self._send_json(404, {"error": "Not found."})
                return

            try:
                body = file_path.read_bytes()
            except OSError:
                LOGGER.exception("Unable to read web asset: %s", file_path.name)
                self._send_json(500, {"error": "Unable to load the chat interface."})
                return

            content_type = mimetypes.guess_type(file_path.name)[0]
            if content_type is None:
                content_type = "application/octet-stream"
            self._send(200, body, f"{content_type}; charset=utf-8")

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self._send_json(404, {"error": "Not found."})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid request length."})
                return

            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "Request is empty or too large."})
                return

            try:
                payload = json.loads(self.rfile.read(content_length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "Request body must be valid JSON."})
                return

            if not isinstance(payload, dict):
                self._send_json(400, {"error": "Request body must be a JSON object."})
                return

            question = payload.get("question")
            medications = payload.get("medications", [])
            if (
                not isinstance(question, str)
                or not question.strip()
                or len(question) > 4_000
            ):
                self._send_json(
                    400,
                    {"error": "Enter a question of 1 to 4,000 characters."},
                )
                return
            if (
                not isinstance(medications, list)
                or len(medications) > 12
                or any(
                    not isinstance(name, str) or len(name) > 200
                    for name in medications
                )
            ):
                self._send_json(
                    400,
                    {"error": "Medication names must be a list of at most 12 strings."},
                )
                return

            user_medications = [name.strip() for name in medications if name.strip()]
            try:
                result = assistant.invoke(
                    {
                        "question": question.strip(),
                        "user_medications": user_medications,
                    }
                )
            except Exception:
                LOGGER.exception("Clinical assistant request failed.")
                self._send_json(
                    502,
                    {
                        "error": (
                            "The assistant could not complete the request. "
                            "Check the server console and try again."
                        )
                    },
                )
                return

            self._send_json(200, _serialize_result(result))

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

    return ClinicalChatHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local clinical RAG chat interface."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit RAGBench train rows for a quick demo; 0 loads all rows.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535.")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    assistant = build_assistant(args.max_rows)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_handler(assistant),
    )
    print(f"Clinical RAG chat available at http://{args.host}:{args.port}")
    print("Research prototype only. Keep the server bound to localhost.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
