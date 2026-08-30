# Apple Ripeness Assessment

A Streamlit application that detects apples in uploaded images or a live camera feed, classifies their ripeness, and generates an assessment report.

## Before you run the application

### 1. Install the prerequisites

Make sure you have:

- Python 3.11 (recommended)
- `pip`
- Git, if you are cloning the repository
- A webcam and browser camera permission if you want to use live detection
- An internet connection if you want to sign in with Google or use Gemini-generated report summaries

### 2. Confirm that the trained model is present

The application expects this file:

```text
Model Training/svm_final_model.pkl
```

It is included in the repository. If it is missing, restore or download it before running the application; ripeness classification will otherwise be unavailable.

### 3. Create and activate a virtual environment

From the repository root, run the commands for your operating system.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You do not need to install TensorFlow for the current application. The included classifier uses scikit-learn and is loaded with Joblib.

### 5. Configure optional services

The application can be explored as a guest without a user account. The following configuration is only required for the related features.

#### Google sign-in and saved analysis history

Create `.streamlit/secrets.toml` and add your Supabase project details:

```toml
[supabase]
url = "https://YOUR_PROJECT_ID.supabase.co"
key = "YOUR_SUPABASE_PUBLISHABLE_KEY"

# Set this when the URL cannot be detected correctly, especially in deployment.
# redirect_url = "http://localhost:8501/"
```

The Supabase project must have Google authentication enabled and be configured with the redirect URL used by the application. Saved history also relies on the `analyses` and `analysis_results` tables and an `apple-images` Storage bucket with suitable row-level security and Storage policies.

Do not commit `.streamlit/secrets.toml`; it is already listed in `.gitignore`.

#### Gemini-generated report summaries

Reports still work without Gemini and will use a local fallback summary. To enable Gemini-generated summaries, set `GEMINI_API_KEY` before starting Streamlit. You may also override the default model with `GEMINI_MODEL`.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
# Optional: $env:GEMINI_MODEL = "YOUR_MODEL_NAME"
```

macOS or Linux:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# Optional: export GEMINI_MODEL="YOUR_MODEL_NAME"
```

#### Live camera on a remote deployment

Local camera use normally works with the built-in STUN configuration. For a deployment behind restrictive NAT or a firewall, set these environment variables using credentials from your TURN provider:

```text
TURN_URL
TURN_USERNAME
TURN_CREDENTIAL
```

## Run the application

Run this command from the repository root while the virtual environment is active:

```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501` if Streamlit does not open it automatically. Choose **Explore as guest** for a setup that does not use Supabase.

## Common setup problems

- **`ModuleNotFoundError`**: activate the virtual environment and reinstall `requirements.txt`.
- **Classifier unavailable**: confirm that `Model Training/svm_final_model.pkl` exists and run Streamlit from the repository root.
- **Google sign-in fails**: check the Supabase URL, publishable key, Google provider, and allowed redirect URL.
- **Camera does not start**: allow camera access in the browser, close other applications using the webcam, and use HTTPS for remote deployments.
- **Gemini warning in a report**: set `GEMINI_API_KEY`, or continue using the built-in fallback summary.
