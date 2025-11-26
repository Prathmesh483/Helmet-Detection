# Traffic AI System (College Project)

## Quick start (local)
1. Create and activate virtualenv
2. `pip install -r requirements.txt`
3. `python download_models.py`  # downloads yolov8n.pt to models/
4. `streamlit run app.py`

## Deploying on Streamlit Cloud / HuggingFace
- Include models/ directory or ensure `download_models.py` can fetch weights (hosting must allow outgoing HTTP).
- Streamlit Cloud often has no GPU; this is CPU-friendly (yolov8n).
- If you saw `UnpicklingError` in logs, delete the corrupted `models/yolov8n.pt` and re-run `download_models.py` or allow ultralytics to auto-download.
