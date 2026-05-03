# VetAI Lite (Fastest Free Deploy)

This is a simplified, single-service version for easy deployment on free platforms.

## Deploy on Render (free)

1. Connect this GitHub repo.
2. Create a **Web Service**.
3. Use repo root as Root Directory.
4. Render will auto-detect:
   - `Procfile` -> `web: gunicorn app:app --bind 0.0.0.0:$PORT`
   - `runtime.txt` -> Python 3.12.8
   - `requirements.txt`

## API

- `GET /` -> health + model status
- `GET /schema` -> feature column order
- `POST /predict`

### Request
```json
{
  "task": "disease",
  "features": [0,1,0,...]
}
```

or

```json
{
  "task": "pregnancy",
  "features": [0,1,0,...],
  "days_since_breeding": 90
}
```

### Response
```json
{
  "prediction": { ... }
}
```

## Notes

- ML logic and model behavior are unchanged (reused from `backend/ml_engine.py`).
- Models must exist in `backend/models/`.