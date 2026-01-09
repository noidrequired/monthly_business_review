# Monthly Business Review Report Builder (Streamlit)

Upload a shipment Excel/CSV, filter the data, view KPIs, and export a slide-style A4 landscape PDF report.

## Deploy (click-only)

### GitHub
1. Sign in to GitHub
2. Create a new **Public** repository
3. Upload these files to the repo:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Commit to the `main` branch

### Streamlit Community Cloud
1. Go to Streamlit Community Cloud and sign in with GitHub
2. Click **New app**
3. Select your repo, branch = `main`, file = `app.py`
4. Click **Deploy**

## Notes
- Blanks in milestone timestamps are treated as "not reported".
- "Timed out" rule:
  - If Current state = COMPLETED and Current state reason = TRACKING_TIMED_OUT:
    - If all three core actual events are present, it is treated as Completed.
    - Otherwise it is Timed Out.
- On-time performance uses latest planned **END** time + allowed latency.
- You can upload a screenshot inside the app to embed in the final PDF.
