import { useMemo, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

/* ---------- Icons (inline SVG, no deps) ---------- */
const IconBrain = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
    <path d="M8.5 6.5C8.5 5.12 9.62 4 11 4h.4c1.05 0 1.98.63 2.36 1.57.24.6.82.99 1.46.99H16c1.38 0 2.5 1.12 2.5 2.5 0 .25-.04.49-.1.72.98.42 1.6 1.38 1.6 2.43 0 1.19-.8 2.21-1.91 2.49.02.12.03.24.03.36 0 1.38-1.12 2.5-2.5 2.5h-.7c-.62 0-1.18.36-1.44.93A2.55 2.55 0 0 1 11.2 22H11c-1.38 0-2.5-1.12-2.5-2.5V6.5Z" stroke="rgba(2,6,23,.85)" strokeWidth="1.5"/>
    <path d="M8.5 9.5a2 2 0 0 0-2 2c0 .86.54 1.6 1.3 1.88-.18.3-.3.65-.3 1.02a2 2 0 0 0 2 2h.3" stroke="rgba(2,6,23,.85)" strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

const IconUpload = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
    <path d="M12 16V7m0 0 3.5 3.5M12 7 8.5 10.5" stroke="rgba(226,232,240,.95)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M7 16.5c-2 0-3.5-1.5-3.5-3.5S5 9.5 7 9.5c.2-2.3 2.1-4 4.5-4 2.1 0 3.9 1.2 4.4 3 2.1.1 3.6 1.8 3.6 4 0 2.4-1.6 4-3.8 4H15" stroke="rgba(148,163,184,.9)" strokeWidth="1.2" strokeLinecap="round"/>
  </svg>
);

const IconPlay = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <path d="M9 18V6l10 6-10 6Z" fill="rgba(2,6,23,.9)"/>
  </svg>
);

/* ---------------------- App ---------------------- */
export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const modelName = useMemo(() => result?.model || "resnet18", [result]);

  const selectFile = (f) => {
    setError("");
    setResult(null);
    setFile(f || null);
    if (!f) {
      setPreview("");
      return;
    }
    setPreview(URL.createObjectURL(f));
  };

  const clearFile = () => selectFile(null);

  const predict = async () => {
    if (!file) {
      setError("Upload an X-ray image first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/predict`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        setError(data?.detail || "Prediction failed.");
        setResult(null);
      } else {
        setResult(data);
      }
    } catch {
      setError("Network error. Is the backend running on 127.0.0.1:8000 ?");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const label = result?.label;
  const conf = result?.confidence ?? 0;
  const normalP = result?.probs?.NORMAL ?? 0;
  const pneuP = result?.probs?.PNEUMONIA ?? 0;

  return (
    <div className="container">
      <div className="shell">

        {/* LEFT COLUMN */}
        <div className="card">
          <div className="card-inner">

            {/* Header */}
            <div className="header">
              <div className="brand">
                <div className="brand-top">
                  <div className="logo"><IconBrain /></div>
                  <div>
                    <div className="title">MedXfer</div>
                    <div className="subtitle">
                      Transfer Learning demo for medical imaging — Chest X-ray Pneumonia Detection
                    </div>
                  </div>
                </div>
              </div>
              <div className="badges">
                <div className="badge"><span className="dot" />FastAPI</div>
                <div className="badge"><span className="dot" />React</div>
                <div className="badge"><span className="dot" />{modelName}</div>
              </div>
            </div>

            {/* Upload + Preview */}
            <div className="grid">
              <div className="panel">
                <div className="panel-title">Upload <span className="hint">PNG / JPG</span></div>

                <label className="drop">
                  <input type="file" accept="image/*" onChange={(e) => selectFile(e.target.files?.[0])}/>
                  <div className="drop-main">
                    <div className="drop-icon"><IconUpload /></div>
                    <div className="drop-text">
                      <b>Click to upload</b>
                      <span>Frontal chest X-rays recommended</span>
                    </div>
                  </div>

                  {file && (
                    <div className="file-pill">
                      <div className="name">{file.name}</div>
                      <button className="icon-btn" onClick={(e)=>{e.preventDefault(); clearFile();}}>Remove</button>
                    </div>
                  )}
                </label>

                <div className="cta-row">
                  <button className="btn btn-primary" onClick={predict} disabled={loading}>
                    {loading ? "Analyzing..." : "Run Analysis"} {!loading && <IconPlay />}
                  </button>
                  <button className="btn btn-secondary" onClick={clearFile} disabled={!file}>
                    Clear
                  </button>
                </div>

                {error && <div className="toast">{error}</div>}
              </div>

              <div className="panel">
                <div className="panel-title">Preview <span className="hint">Local</span></div>
                <div className={`preview ${loading ? "scanning" : ""}`}>
                  {preview ? <img src={preview} alt="X-ray preview" /> :
                    <div className="preview-empty">Upload an image to preview here</div>}
                  <div className="scanline" />
                </div>
                <div className="footer-note">Educational demo only. Not a medical device.</div>
              </div>
            </div>
          </div>
        </div>

        {/* TRANSFER LEARNING EXPLANATION (BIG) */}
        <div className="card tl-card">
          <div className="card-inner">
            <h2 className="tl-title">How Transfer Learning Is Applied</h2>

            <p className="tl-text">
              This system uses <b>Transfer Learning</b> to build a medical image classifier efficiently and reliably.
              Training a deep CNN from scratch on limited medical data is impractical, so we reuse knowledge from a
              model trained on a large-scale dataset.
            </p>

            <p className="tl-text">
              We start with a <b>ResNet backbone pre-trained on ImageNet</b>. ImageNet enables the network to learn
              universal visual features such as edges, textures, gradients, and spatial hierarchies.
            </p>

            <p className="tl-text">
              The original classification head is <b>replaced</b> with a new layer specific to the medical task
              (<b>Normal vs Pneumonia</b>). During training:
            </p>

            <ul className="tl-list">
              <li>Early convolutional layers are <b>frozen</b> to preserve general visual knowledge</li>
              <li>The new classifier head is trained on X-ray images</li>
              <li>Final layers are <b>fine-tuned</b> to adapt to medical patterns</li>
            </ul>

            <p className="tl-text highlight">
              Transfer Learning reduces training time, improves generalization on small datasets,
              and significantly lowers overfitting — making it ideal for healthcare AI applications.
            </p>
          </div>
        </div>

        {/* RIGHT COLUMN — RESULTS */}
        <div className="side">
          <div className="card">
            <div className="card-inner">
              <div className="panel-title">Results <span className="hint">Test AUROC ≈ 0.94</span></div>

              {!result ? (
                <div className="disclaimer">
                  Upload an X-ray and click <b>Run Analysis</b> to see predictions and probabilities.
                </div>
              ) : (
                <div className="big">
                  <div className="diagnosis">
                    <div className={`label ${label === "PNEUMONIA" ? "bad" : "good"}`}>{label}</div>
                    <div className="conf">
                      Confidence<br/><b>{(conf * 100).toFixed(2)}%</b>
                    </div>
                  </div>

                  <div className="meter">
                    <div style={{ width: `${conf * 100}%` }} />
                  </div>

                  <div className="bars">
                    <div>
                      <div className="row"><div className="k">NORMAL</div><div className="v">{(normalP*100).toFixed(2)}%</div></div>
                      <div className="track"><div className="fill" style={{ width: `${normalP*100}%` }} /></div>
                    </div>
                    <div>
                      <div className="row"><div className="k">PNEUMONIA</div><div className="v">{(pneuP*100).toFixed(2)}%</div></div>
                      <div className="track"><div className="fill" style={{ width: `${pneuP*100}%` }} /></div>
                    </div>
                  </div>

                  <div className="disclaimer">{result.disclaimer}</div>
                </div>
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
