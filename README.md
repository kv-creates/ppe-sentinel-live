# PPE Sentinel Live — Cloud Demo

Standalone, one-click-deployable demo of the PPE non-compliance detector.
YOLOv8 inference runs inside the app itself — no backend server needed.

Live app (after deploy): `https://<your-app>.streamlit.app`
Full project (training, API, Windows 7 UI, report):
`https://github.com/kv-creates/ppe-detector`

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to GitHub as `ppe-sentinel-live` (see commands below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. New app → repository `kv-creates/ppe-sentinel-live`, branch `main`,
   main file `app.py` → Deploy.
4. First load takes about a minute (model download into memory happens once
   from the bundled `model/best.pt`).

## Run locally

```bat
pip install -r requirements.txt
streamlit run app.py
```

## Video + Telegram alerts

- Source menu includes **Upload video (MP4)**: every 5th frame is scored,
  violations are counted, and an annotated MP4 plays back in the page.
- **Telegram:** video violations in the **red zone only** message your bot.
  Add repository Secrets in Streamlit Cloud (App settings → Secrets) with:

```toml
[telegram]
bot_token = "123456:ABC..."
chat_id = "987654321"
```

  Video violations then message your bot automatically. Locally, the same
  keys work via `st secrets.toml` or the main project's environment
  variables.

## Contents

```text
ppe-sentinel-live/
├── app.py            # Standalone Streamlit app (detect + counters + alerts)
├── win7.css          # Windows 7 Aero theme (shared look with main project)
├── model/best.pt     # YOLOv8m PPE weights, 52 MB (60 epochs, T4 GPU)
├── samples/          # 4 demo site photos
└── requirements.txt  # Slim cloud dependencies (CPU torch via ultralytics)
```

## Model

YOLOv8m trained 60 epochs on a T4 GPU on 1,331 real construction-site photos
(6 classes: person, helmet, vest, no-helmet, no-vest, boots). Held-out test:
mAP@0.5 0.70, precision 0.72, recall 0.70. Known weakness: no-helmet recall
is low (small bare heads, 282 training samples) — see the main repository for
the full evaluation, KPIs, and the plan to fix it with more violation photos.

## License

MIT — see the main repository.
