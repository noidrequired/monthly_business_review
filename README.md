# Monthly Business Review Report Builder (Streamlit)

Upload a shipment Excel/CSV, filter the data, view KPIs/tables, and export a slide-style A4 landscape PDF.

## Deploy (click-only)

### GitHub (web)
1. Sign in to GitHub
2. Create a new **Public** repository
3. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. (Optional) Add logo file:
   - `assets/p44_logo.png`
5. Commit to `main`

### Streamlit Community Cloud (web)
1. Go to Streamlit Community Cloud and sign in with GitHub
2. Click **New app**
3. Select repo, branch = `main`, file = `app.py`
4. Click **Deploy**

## Notes
- Blanks in milestone timestamps are treated as "not reported".
- Timed out rule:
  - If Current state = COMPLETED and Current state reason = TRACKING_TIMED_OUT:
    - If all three core actual events are present → Completed
    - Else → Timed Out
- OTP uses latest planned END time + allowed latency.
- PDF uses auto-fit tables: font/padding/column widths adjust to fit the page.
- Embedded images: upload multiple; each becomes a separate PDF page with header & subtitle.

