"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { API_URL, fetchHealth, fetchSchema, predict } from "@/lib/api";
import { diseaseColumnLabel, normalizeSymptomKey, pregnancyColumnLabel } from "@/lib/labels";

export default function Home() {
  const [health, setHealth] = useState(null);
  const [schemaError, setSchemaError] = useState(null);
  const [tab, setTab] = useState("disease");

  const [diseaseCols, setDiseaseCols] = useState([]);
  const [pregCols, setPregCols] = useState([]);

  const [diseaseChecked, setDiseaseChecked] = useState({});
  const [pregChecked, setPregChecked] = useState({});
  const [daysSinceBreeding, setDaysSinceBreeding] = useState(0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, s] = await Promise.all([fetchHealth(), fetchSchema()]);
        if (cancelled) return;
        setHealth(h);
        const d = s.disease_symptom_columns || [];
        const p = s.pregnancy_feature_columns || s.pregnancy_checklist_fallback_order || [];
        setDiseaseCols(d);
        setPregCols(p);
        setDiseaseChecked(Object.fromEntries(d.map((c) => [normalizeSymptomKey(c), false])));
        setPregChecked(Object.fromEntries(p.map((c) => [c, false])));
      } catch (e) {
        if (!cancelled) setSchemaError(e.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const diseaseFeatures = useMemo(() => {
    return diseaseCols.map((col) => (diseaseChecked[normalizeSymptomKey(col)] ? 1 : 0));
  }, [diseaseCols, diseaseChecked]);

  const pregnancyFeatures = useMemo(() => {
    return pregCols.map((col) => (pregChecked[col] ? 1 : 0));
  }, [pregCols, pregChecked]);

  const toggleDisease = useCallback((key) => {
    setDiseaseChecked((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const togglePreg = useCallback((key) => {
    setPregChecked((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const runDisease = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const res = await predict("disease", diseaseFeatures);
      setPrediction(res.prediction);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const runPregnancy = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const res = await predict("pregnancy", pregnancyFeatures, Number(daysSinceBreeding) || 0);
      setPrediction(res.prediction);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <h1>VetAI — Veterinary assistant</h1>
      <p className="muted">
        Cattle-focused checklist UI. Predictions are computed on your backend (<code>{API_URL}</code>).
      </p>
      <p className="disclaimer">
        This is an AI-assisted educational tool. It is not a medical diagnosis. Always consult a licensed
        veterinarian for examination, testing, and treatment.
      </p>

      {schemaError && <p className="error">API: {schemaError}</p>}
      {health && (
        <p className="status">
          Backend status: disease model {health.disease_model ? "ready" : "missing"}, pregnancy{" "}
          {health.pregnancy_model ? "ready" : "missing"}, knowledge {health.knowledge_base ? "ready" : "missing"}.
        </p>
      )}

      <div className="tabs">
        <button type="button" className={tab === "disease" ? "active" : ""} onClick={() => setTab("disease")}>
          Illness checklist
        </button>
        <button type="button" className={tab === "pregnancy" ? "active" : ""} onClick={() => setTab("pregnancy")}>
          Pregnancy helper
        </button>
      </div>

      {tab === "disease" && (
        <div className="panel">
          <p className="muted">Check signs that fit. Order matches your trained model columns from GET /schema.</p>
          <div className="checkgrid">
            {diseaseCols.map((col) => {
              const key = normalizeSymptomKey(col);
              return (
                <label key={col}>
                  <input
                    type="checkbox"
                    checked={!!diseaseChecked[key]}
                    onChange={() => toggleDisease(key)}
                  />
                  {diseaseColumnLabel(col)}
                </label>
              );
            })}
          </div>
          <div className="actions">
            <button type="button" className="primary" disabled={loading || !diseaseCols.length} onClick={runDisease}>
              {loading ? "…" : "Get diagnosis"}
            </button>
          </div>
        </div>
      )}

      {tab === "pregnancy" && (
        <div className="panel">
          <p className="muted">Signs and days since breeding are sent as JSON to POST /predict.</p>
          <div className="actions">
            <label className="muted">
              Days since breeding:{" "}
              <input
                type="number"
                min={0}
                max={400}
                value={daysSinceBreeding}
                onChange={(e) => setDaysSinceBreeding(e.target.value)}
              />
            </label>
          </div>
          <div className="checkgrid">
            {pregCols.map((col) => (
              <label key={col}>
                <input type="checkbox" checked={!!pregChecked[col]} onChange={() => togglePreg(col)} />
                {pregnancyColumnLabel(col)}
              </label>
            ))}
          </div>
          <div className="actions">
            <button type="button" className="primary" disabled={loading || !pregCols.length} onClick={runPregnancy}>
              {loading ? "…" : "Estimate stage"}
            </button>
          </div>
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {prediction && (
        <div className="result panel">
          <h3>Prediction</h3>
          <pre className="json">{JSON.stringify(prediction, null, 2)}</pre>
        </div>
      )}
    </main>
  );
}
