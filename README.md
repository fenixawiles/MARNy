🧠 MARNy — Multi-Agent Recursive Network

MARNy is a lightweight Flask application that demonstrates recursive, autonomous document refinement through iterative AI dialogue.
It serves as a proof-of-concept for Recursive Cognition in Practice (RCIP) — showing that an AI system can critique, revise, and self-evaluate until it converges on a rigorously stable output.

🚀 Overview

MARNy runs a continuous critique–revision–evaluation loop using OpenAI’s GPT-4o-mini model.
Each iteration:

Critiques the current document.

Generates a revision.

Evaluates whether further refinement is needed.

The process stops automatically once no substantive issues remain, logging every iteration for audit and validation.

🧩 Architecture

Core Files

app.py — Flask backend; handles routing, OpenAI client setup, and recursive loop logic.

templates/index.html — Frontend UI; minimal, single-page design for submitting text and visualizing loop history.

test_env.py — Environment validation script ensuring API key detection and dependency integrity.

startup.log — Example run log with initialization diagnostics and environment checks.

⚙️ Requirements

Python 3.13+

Install dependencies:

pip install flask openai python-dotenv

🔑 Environment Setup

Create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here


If this variable is missing, the app will log warnings such as:

[WARNING] OPENAI_API_KEY is not set. Critique requests will fail until it is configured.

🧠 How It Works

User Input: Paste or upload a document into the MARNy interface.

Loop Initialization: app.py passes the text to the OpenAI client.

Autonomous Recursion:

Each cycle performs a Critique → Revision → Evaluation sequence.

Results are rendered dynamically in the browser and saved to an audit log.

Completion:

MARNy halts when no new substantive changes are detected.

The final version and all prior loops remain visible.

This architecture embodies recursive cognition: MARNy doesn’t just edit — it thinks through revision.

🧪 Testing

Run environment diagnostics:

python test_env.py


You should see outputs confirming:

Python, Flask, and OpenAI SDK versions

.env detection

Key length validation

Successful client initialization

Sample (from startup.log):

[INFO]   Flask version: 3.1.2
[INFO]   OpenAI SDK version: 2.2.0
[INFO]   Detected OPENAI_API_KEY with length 164.
[INFO]   OpenAI client initialized successfully.
[INFO]   Starting Flask development server on http://localhost:5000 ...

🖥️ Running MARNy

Start the app:

python app.py


Then visit:

http://localhost:5000


You’ll see the MARNy Recursive Review interface.
Paste your text → click Start Review → watch MARNy iterate autonomously.

🪶 Conceptual Basis

MARNy operationalizes a core insight from Recursive Cognition in Practice (Wiles, 2024):

“Rigor emerges not from external validation, but from internal recursion.”

This app is an MVP demonstration of Recursive Rigor, Reflexive Validation, and Multi-Model Autonomy — principles later formalized in TRP (The Recursive Protocol).
