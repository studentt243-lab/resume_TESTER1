# 📄 ATS Resume Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) compatibility and tells you exactly how to improve it, powered by Google Gemini Flash.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- **ATS score (0-100)** = 60% Gemini AI review + 40% deterministic rule-based checks
- Optional **job description** input for keyword-gap analysis
- Prioritised improvements (High / Medium / Low), strengths and weaknesses
- Before/after **bullet-point rewrites**
- Shows the **text an ATS actually extracts** from your file
- Download the full report as Markdown

## How the score works

| Part | Weight | What it measures |
|------|--------|------------------|
| Rule-based checks | 40% | Email/phone/links, standard section headings, length, bullets, action verbs, quantified results, dates (+ literal keyword match if a job description is given) |
| Gemini AI review | 60% | Keywords & relevance, formatting & structure, content & impact, readability, completeness |

The score is an **estimate**. Real ATS products (Workday, Greenhouse, Lever, ...) differ, and no tool can give you the exact number they use.

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Get a free API key at <https://aistudio.google.com/apikey> and either paste it in the sidebar or set it up as below.

### Configure the API key (optional)

Create `.streamlit/secrets.toml` (already in `.gitignore`):

```toml
GEMINI_API_KEY = "your-key-here"
```

or set an environment variable: `export GEMINI_API_KEY=your-key-here`

### Choose the model

The model id is editable in the sidebar, or set `GEMINI_MODEL` (env var). Google renames and retires models periodically, so check the current Flash model id at <https://ai.google.dev/gemini-api/docs/models>. If the chosen model returns "not found", the app automatically tries the fallbacks listed in `FALLBACK_MODELS` at the top of `app.py`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with access granted to Streamlit).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → select your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. You get a public URL like `https://<app-name>.streamlit.app`.

> ⚠️ If you put *your* key in Secrets, anyone using your public app spends *your* quota. To avoid this, leave the secret empty and let users paste their own key in the sidebar.

## Limitations

- Scanned / image-only PDFs have no readable text. Most ATS systems can't read them either, and the app will tell you.
- Resume text is sent to the Gemini API. Don't upload resumes you aren't comfortable sharing with Google's API terms.
- AI feedback can be wrong. Treat it as suggestions, never invent metrics.

## Project structure

```
app.py             # Streamlit UI + scoring logic + Gemini call
requirements.txt   # Python dependencies
README.md
```

## License

MIT# resume_TESTER1
