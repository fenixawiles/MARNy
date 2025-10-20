import os
import platform
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, render_template, request, __version__ as flask_version
from openai import OpenAI, __version__ as openai_version

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

_client: Optional[OpenAI] = None
_api_key: Optional[str] = None
_client_ready: bool = False
_client_error: Optional[str] = None
_startup_events: List[Dict[str, str]] = []

REVIEW_PROMPT = (
    "You are a rigorous peer reviewer applying The Recursive Protocol (TRP). "
    "Focus ONLY on substantive issues:\n"
    "- Methodological gaps or logical flaws\n"
    "- Missing evidence or unsupported claims\n"
    "- Unclear core arguments\n"
    "- Bias or circular reasoning\n\n"
    "- Do NOT implement citations or information that you cannot verify with your current knowledge."
    "DO NOT critique:\n"
    "- Minor wording choices or semantic phrasing\n"
    "- Stylistic preferences\n"
    "- Issues already addressed in prior revisions\n\n"
    "If the document is methodologically sound, state: 'No substantive issues remain.'"
    
)

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not _api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Unable to create OpenAI client."
            )
        _client = OpenAI(api_key=_api_key)
    return _client


def record_startup_event(message: str, level: str = "info") -> None:
    ensure_audit_directory()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized_level = level.lower()
    prefix = {"info": "INFO", "warning": "WARNING", "error": "ERROR"}.get(
        normalized_level,
        "INFO",
    )
    line = f"{prefix}: {message}"
    print(line, flush=True)

    _startup_events.append({"level": normalized_level, "message": message})

    log_path = os.path.join("audit_trails", "startup.log")
    with open(log_path, "a", encoding="utf-8") as startup_log:
        startup_log.write(f"{timestamp} [{prefix}] {message}\n")


def get_startup_messages() -> List[Dict[str, str]]:
    return [event for event in _startup_events if event["level"] != "info"]


def inspect_env_file(env_path: str) -> None:
    if not os.path.isfile(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            lines = env_file.read().splitlines()
    except OSError as exc:  # noqa: PERF203 - user feedback is more important here
        record_startup_event(
            f"Could not read .env file for validation: {exc}",
            level="warning",
        )
        return

    for index, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "=" not in stripped:
            record_startup_event(
                (
                    f"Line {index} of .env has no '=': '{stripped}'. If this was "
                    "intended to continue the OPENAI_API_KEY, merge it back onto a "
                    "single line."
                ),
                level="warning",
            )
            continue

        key, value = stripped.split("=", 1)
        if key != "OPENAI_API_KEY":
            continue

        if not value:
            record_startup_event(
                "OPENAI_API_KEY is defined but empty in the .env file.",
                level="warning",
            )

        if len(value) < 50:
            record_startup_event(
                (
                    "OPENAI_API_KEY looks shorter than expected. Ensure the full "
                    "key is present on one line in the .env file."
                ),
                level="warning",
            )

        next_index = index
        if next_index < len(lines):
            next_line = lines[next_index].strip()
            if next_line and "=" not in next_line and not next_line.startswith("#"):
                record_startup_event(
                    (
                        "Detected additional text on the line after OPENAI_API_KEY. "
                        "This usually means the key was split across multiple lines."
                    ),
                    level="warning",
                )


def generate_critique(document_text: str) -> str:
    response = get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": document_text},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_revision(document_text: str, critique: str) -> str:
    """Generate a revised document based on the critique."""
    client = get_client()
    revision_prompt = (
        "You are revising a document based on peer review feedback. "
        "Rewrite the document to address all critique points while preserving "
        "the core content and intent. Return only the revised document text."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": revision_prompt},
            {
                "role": "user",
                "content": (
                    f"Original Document:\n{document_text}\n\n"
                    f"Critique:\n{critique}\n\n"
                    "Provide the revised document:"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def should_continue_refinement(
    current_critique: str, previous_critique: str, iteration: int
) -> Tuple[bool, str]:
    """
    Determine if another refinement loop is warranted.
    Returns (should_continue, reason_for_stopping)
    """
    if iteration >= 10:
        return False, "Maximum safety limit reached (10 iterations)"

    if iteration == 1:
        return True, ""

    eval_prompt = (
        "You are evaluating whether peer review feedback represents substantive "
        "methodological concerns or has devolved into nitpicking and semantic quibbling. "
        "Respond with ONLY 'SUBSTANTIVE' or 'NITPICKING'.\n\n"
        f"Previous critique:\n{previous_critique}\n\n"
        f"Current critique:\n{current_critique}\n\n"
        "Has the critique shifted from addressing real methodological gaps to "
        "nitpicking minor semantic issues or restating previous points?"
    )

    response = get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0,
    )

    evaluation = response.choices[0].message.content.strip().upper()

    if "NITPICKING" in evaluation:
        return False, "Critique devolved into nitpicking rather than substantive feedback"

    return True, ""


def ensure_audit_directory() -> None:
    os.makedirs("audit_trails", exist_ok=True)

def append_loop_to_audit_log(
    log_filename: str,
    iteration: int,
    document_text: str,
    critique: str,
    revision: str,
    evaluation: str,
) -> None:
    ensure_audit_directory()
    log_path = os.path.join("audit_trails", log_filename)
    header = f"Loop {iteration}\n"
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(header)
        log_file.write(f"Input Document:\n{document_text}\n\n")
        log_file.write(f"Critique:\n{critique}\n\n")
        log_file.write(f"Revision:\n{revision}\n\n")
        log_file.write(f"Stopping evaluation: {evaluation}\n\n")


def append_summary_to_audit_log(
    log_filename: str, total_loops: int, stopping_reason: str
) -> None:
    ensure_audit_directory()
    log_path = os.path.join("audit_trails", log_filename)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("---\n")
        log_file.write(f"Total loops completed: {total_loops}\n")
        if stopping_reason:
            log_file.write(f"Stopping reason: {stopping_reason}\n")
        log_file.write("\n")


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        document_text="",
        loops=[],
        final_document="",
        log_file="",
        stop_reason="",
        refinement_complete=False,
        error_message=None,
        startup_messages=get_startup_messages(),
        client_error=_client_error,
    )


@app.route("/critique", methods=["POST"])
def critique():
    document_text = request.form.get("document_text", "").strip()

    error_message = None
    loops: List[Dict[str, str]] = []
    final_document = document_text
    stop_reason = ""
    refinement_complete = False
    log_file = ""

    if not document_text:
        error_message = "Please provide document text for critique."
    elif not _client_ready:
        error_message = (
            "The OpenAI client could not be initialized at startup, so critiques "
            "cannot be generated. Check the startup diagnostics below for the "
            "recorded error and verify your OPENAI_API_KEY."
        )
    else:
        log_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        current_document = document_text
        previous_critique = ""
        iteration = 1
        summary_reason = ""

        try:
            while True:
                critique_text = generate_critique(current_document)
                loop_entry: Dict[str, str] = {
                    "iteration": str(iteration),
                    "document": current_document,
                    "critique": critique_text,
                    "revision": current_document,
                    "evaluation": "",
                }
                loops.append(loop_entry)

                if "no substantive issues remain" in critique_text.lower():
                    stop_reason = "Reviewer indicated no substantive issues remain."
                    loop_entry["evaluation"] = stop_reason
                    append_loop_to_audit_log(
                        log_file,
                        iteration,
                        current_document,
                        critique_text,
                        current_document,
                        stop_reason,
                    )
                    final_document = current_document
                    refinement_complete = True
                    summary_reason = stop_reason
                    break

                continue_refinement, reason = should_continue_refinement(
                    critique_text, previous_critique, iteration
                )

                if not continue_refinement:
                    stop_reason = reason or "Stopping conditions met."
                    loop_entry["evaluation"] = stop_reason
                    append_loop_to_audit_log(
                        log_file,
                        iteration,
                        current_document,
                        critique_text,
                        current_document,
                        stop_reason,
                    )
                    final_document = current_document
                    refinement_complete = True
                    summary_reason = stop_reason
                    break

                try:
                    revision_text = generate_revision(current_document, critique_text)
                except Exception as exc:  # noqa: BLE001
                    error_message = (
                        "An error occurred while generating the revision: "
                        f"{exc}"
                    )
                    stop_reason = "Stopped due to revision generation error."
                    loop_entry["evaluation"] = stop_reason
                    append_loop_to_audit_log(
                        log_file,
                        iteration,
                        current_document,
                        critique_text,
                        current_document,
                        stop_reason,
                    )
                    summary_reason = stop_reason
                    break

                evaluation_message = "Continuing refinement (substantive issues remain)."
                loop_entry["revision"] = revision_text
                loop_entry["evaluation"] = evaluation_message
                append_loop_to_audit_log(
                    log_file,
                    iteration,
                    current_document,
                    critique_text,
                    revision_text,
                    evaluation_message,
                )

                previous_critique = critique_text
                current_document = revision_text
                final_document = revision_text
                iteration += 1

            if loops:
                append_summary_to_audit_log(
                    log_file,
                    len(loops),
                    summary_reason or stop_reason,
                )
        except Exception as exc:  # noqa: BLE001
            error_message = f"An error occurred while generating the critique: {exc}"
            summary_reason = "Stopped due to critique generation error."
            if loops:
                loops[-1]["evaluation"] = summary_reason
                append_summary_to_audit_log(log_file, len(loops), summary_reason)

    return render_template(
        "index.html",
        document_text="" if not error_message else document_text,
        loops=loops,
        final_document=final_document,
        log_file=log_file,
        stop_reason=stop_reason,
        refinement_complete=refinement_complete,
        error_message=error_message,
        startup_messages=get_startup_messages(),
        client_error=_client_error,
    )


def main() -> None:
    global _client_ready, _client_error, _api_key

    _startup_events.clear()
    _client_error = None
    _client_ready = False

    record_startup_event("Environment diagnostics:")
    record_startup_event(f"  Python executable: {sys.executable}")
    record_startup_event(f"  Python version: {platform.python_version()}")
    record_startup_event(f"  Flask version: {flask_version}")
    record_startup_event(f"  OpenAI SDK version: {openai_version}")
    record_startup_event(f"  Template folder: {TEMPLATE_DIR}")

    index_template_path = os.path.join(TEMPLATE_DIR, "index.html")
    if os.path.isfile(index_template_path):
        record_startup_event(f"  Found index.html template at {index_template_path}")
    else:
        record_startup_event(
            "index.html template is missing. Ensure the templates directory was copied in full.",
            level="error",
        )

    env_path = os.path.abspath(".env")
    env_status = "found" if os.path.isfile(env_path) else "missing"
    record_startup_event(f"  .env path: {env_path} ({env_status})")

    inspect_env_file(env_path)

    load_result = load_dotenv(dotenv_path=env_path)
    record_startup_event(
        "  Loaded environment variables from .env." if load_result else "  No new variables loaded from .env.",
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        _api_key = None
        record_startup_event(
            "OPENAI_API_KEY is not set. Critique requests will fail until it is configured.",
            level="warning",
        )
    else:
        api_key = api_key.strip()
        _api_key = api_key
        if "\n" in api_key or "\r" in api_key:
            record_startup_event(
                "OPENAI_API_KEY contains newline characters. Ensure the key is on a single line in the .env file.",
                level="warning",
            )
        if not api_key.startswith("sk-"):
            record_startup_event(
                "OPENAI_API_KEY does not start with 'sk-'. Double-check that the correct key was pasted.",
                level="warning",
            )
        if len(api_key) < 50:
            record_startup_event(
                "OPENAI_API_KEY appears shorter than expected. Confirm the entire key was copied.",
                level="warning",
            )
        if api_key:
            record_startup_event(f"  Detected OPENAI_API_KEY with length {len(api_key)}.")
        else:
            record_startup_event(
                "OPENAI_API_KEY is defined but empty after trimming whitespace.",
                level="warning",
            )
            _api_key = None

    record_startup_event("Attempting to initialize OpenAI client...")
    if not _api_key:
        _client_ready = False
        _client_error = (
            "OPENAI_API_KEY was not available when initialization was attempted."
        )
        record_startup_event(
            "Skipped OpenAI client initialization because the API key is missing.",
            level="error",
        )
    else:
        try:
            get_client()
            record_startup_event("OpenAI client initialized successfully.")
            _client_ready = True
        except Exception as exc:  # noqa: BLE001
            _client_ready = False
            _client_error = str(exc)
            record_startup_event(
                f"Failed to initialize OpenAI client: {exc}", level="error"
            )

    record_startup_event("Starting Flask development server on http://localhost:5000 ...")
    try:
        app.run(debug=True, use_reloader=False)
    except Exception as exc:  # noqa: BLE001
        record_startup_event(f"Flask exited with an error: {exc}", level="error")
        raise
    finally:
        record_startup_event("Flask development server has stopped.")


if __name__ == "__main__":
    main()