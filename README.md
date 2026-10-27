# AI-Based Adaptive Interview Preparation System

**Live demo:** https://ai-interview-preparation-cb5tdxnivdynztq8tfhxeh.streamlit.app/

This project demonstrates an adaptive technical interview preparation workflow using classical AI concepts.

## Project contents

- `desktop_prototype.py` — supplied Python/Tkinter prototype source code.
- `AI_Interview_Preparation_Presentation.pptx` — 20-slide project presentation.
- `AI_Interview_Preparation_Project_Report.docx` — project report.
- `screenshots/` — screenshots supplied with the project.
- `requirements.txt` — no third-party packages are required for the Tkinter prototype.

## AI concepts covered

- Classical search: BFS, DFS, Uniform Cost Search, Iterative Deepening, Bidirectional Search.
- Informed and local search: Greedy Best-First, A*, Hill Climbing, Simulated Annealing.
- Constraint satisfaction: domain filtering, constraint relaxation and backtracking.
- Propositional logic: forward chaining, backward chaining and resolution.
- First-order logic: unification and forward chaining.
- Question evaluation based on expected keywords, plus an offline fallback question generator.

## Run the supplied desktop prototype

The source file uses Python's standard library and Tkinter. On a desktop installation of Python that includes Tkinter:

```bash
python desktop_prototype.py
```

## Gemini API key safety

The source is configured to read an optional Gemini API key from the `GEMINI_API_KEY` environment variable. Do not commit API keys or other secrets to GitHub.

Windows PowerShell example:

```powershell
$env:GEMINI_API_KEY="YOUR_NEW_API_KEY"
python desktop_prototype.py
```

If an API key was previously embedded in a source file or shared, revoke it and create a replacement before publishing the repository.

## Note about the live demo

The live link above is a Streamlit-hosted application. The source file supplied for this repository is a Tkinter desktop prototype; it is not the Streamlit deployment source. To make this repository fully reproducible for the hosted Streamlit app, add the Streamlit app's own `app.py` and any associated dependency/configuration files if they are available.
