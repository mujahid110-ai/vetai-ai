export const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://YOUR_BACKEND_URL";

/**
 * POST /predict — body matches production Flask API.
 * @param {"disease"|"pregnancy"} task
 * @param {number[]} features — 0/1 flags in model column order (see GET /schema)
 * @param {number} [daysSinceBreeding] — pregnancy only
 */
export async function predict(task, features, daysSinceBreeding = 0) {
  const body = { task, features };
  if (task === "pregnancy") {
    body.days_since_breeding = daysSinceBreeding;
  }
  const res = await fetch(`${API_URL.replace(/\/$/, "")}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

export async function fetchSchema() {
  const base = API_URL.replace(/\/$/, "");
  const res = await fetch(`${base}/schema`);
  if (!res.ok) {
    throw new Error("Could not load model schema from API.");
  }
  return res.json();
}

export async function fetchHealth() {
  const base = API_URL.replace(/\/$/, "");
  const res = await fetch(`${base}/`);
  if (!res.ok) {
    throw new Error("Health check failed.");
  }
  return res.json();
}
