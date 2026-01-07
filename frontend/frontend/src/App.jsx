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

function ImageModal({ open, title, src, onClose }) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">{title}</div>
          <button className="btn-lite" onClick={onClose}>Close</button>
        </div>
        <div className="modal-body">
          <img className="modal-img" src={src} alt={title} />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const [activeTab, setActiveTab] = useState("original"); // original | gradcam

  const [modalOpen, setModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState("");
  const [modalSrc, setModalSrc] = useState("");

  const modelName = useMemo(() => result?.model || "resnet18", [result]);

  const gradcamB64 = result?.gradcam_overlay_png_b64 || "";
  const gradcamSrc = gradcamB64 ? `data:image/png;base64,${gradcamB64}` : "";

  const openModal = (title, src) => {
    if (!src) return;
    setModalTitle(title);
    setModalSrc(src);
    setModalOpen(true);
  };

  const selectFile = (f) => {
    setError("");
    setResult(null);
    setFile(f || null);
    if (!f) return setPreview("");
    setPreview(URL.createObjectURL(f));
  };

  const clearFile = () => {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  };

  const predict = async () => {
    if (!file) return setError("Upload an X-ray image first.");
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
        setActiveTab("gradcam"); // auto show XAI after prediction
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
    <div className="page">
      <div className="wrap">

        {/* HEADER */}
        <div className="topbar">
          <div className="brand">
            <div className="logo"><IconBrain /></div>
            <div>
              <div className="title">MedXfer</div>
              <div className="subtitle">
                Chest X-ray Pneumonia Detection (Transfer Learning) + Explainability (Grad-CAM)
              </div>
            </div>
          </div>

          <div className="badges">
            <span className="badge">FastAPI</span>
            <span className="badge">React</span>
            <span className="badge">{modelName}</span>
          </div>
        </div>

        {/* MAIN LAYOUT: natural breakpoints via min widths */}
        <div className="layout">

          {/* LEFT: Upload + Images */}
          <div className="card">
            <div className="card-body">

              <div className="row-grid">
                {/* Upload */}
                <div className="panel upload">
                  <div className="panel-title">Upload <span className="hint">PNG/JPG</span></div>

                  <label className="drop">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => selectFile(e.target.files?.[0])}
                    />
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
                        <button className="btn-lite" onClick={(e) => { e.preventDefault(); clearFile(); }}>
                          Remove
                        </button>
                      </div>
                    )}
                  </label>

                  <div className="actions">
                    <button className="btn primary" onClick={predict} disabled={loading}>
                      {loading ? "Analyzing..." : "Run Analysis"} {!loading && <IconPlay />}
                    </button>
                    <button className="btn ghost" onClick={clearFile} disabled={!file && !result}>
                      Clear
                    </button>
                  </div>

                  {error && <div className="alert">{error}</div>}
                  <div className="footnote">Educational demo only. Not a medical device.</div>
                </div>

                {/* Images area */}
                <div className="panel images">
                  <div className="panel-title">
                    Images <span className="hint">Click to expand</span>
                  </div>

                  {/* Tabs appear on smaller screens; on large they still help UX */}
                  <div className="tabs">
                    <button
                      className={`tab ${activeTab === "original" ? "active" : ""}`}
                      onClick={() => setActiveTab("original")}
                    >
                      Original
                    </button>
                    <button
                      className={`tab ${activeTab === "gradcam" ? "active" : ""}`}
                      onClick={() => setActiveTab("gradcam")}
                      disabled={!gradcamSrc}
                      title={!gradcamSrc ? "Run analysis first" : "Grad-CAM"}
                    >
                      Grad-CAM
                    </button>
                  </div>

                  {activeTab === "original" ? (
                    <button
                      className={`media ${loading ? "scanning" : ""}`}
                      onClick={() => openModal("Original X-ray", preview)}
                      disabled={!preview}
                    >
                      {preview ? (
                        <img src={preview} alt="Original X-ray" />
                      ) : (
                        <div className="empty">Upload an image to preview here</div>
                      )}
                      <div className="scanline" />
                    </button>
                  ) : (
                    <button
                      className="media"
                      onClick={() => openModal("Grad-CAM Overlay", gradcamSrc)}
                      disabled={!gradcamSrc}
                    >
                      {gradcamSrc ? (
                        <img src={gradcamSrc} alt="Grad-CAM overlay" />
                      ) : (
                        <div className="empty">Run analysis to generate Grad-CAM</div>
                      )}
                    </button>
                  )}

                  <div className="xai-note">
                    Grad-CAM highlights regions that influenced the prediction.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: Results */}
          <div className="card">
            <div className="card-body">
              <div className="panel-title">Results <span className="hint">Clinical-style metrics</span></div>

              {!result ? (
                <div className="muted-block">
                  Upload an X-ray and click <b>Run Analysis</b>.
                </div>
              ) : (
                <>
                  <div className="result-top">
                    <div className={`result-label ${label === "PNEUMONIA" ? "bad" : "good"}`}>{label}</div>
                    <div className="result-conf">
                      Confidence<br/><b>{(conf * 100).toFixed(2)}%</b>
                    </div>
                  </div>

                  <div className="meter">
                    <div style={{ width: `${conf * 100}%` }} />
                  </div>

                  <div className="bars">
                    <div className="bar">
                      <div className="bar-row"><span>NORMAL</span><span>{(normalP * 100).toFixed(2)}%</span></div>
                      <div className="track"><div className="fill" style={{ width: `${normalP * 100}%` }} /></div>
                    </div>
                    <div className="bar">
                      <div className="bar-row"><span>PNEUMONIA</span><span>{(pneuP * 100).toFixed(2)}%</span></div>
                      <div className="track"><div className="fill" style={{ width: `${pneuP * 100}%` }} /></div>
                    </div>
                  </div>

                  <div className="small">{result.disclaimer}</div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* EXPLANATION */}
        <div className="card">
          <div className="card-body">
            <h2 className="section-title">How Transfer Learning Is Applied</h2>
            <p className="p">
              We reuse a <b>ResNet backbone pre-trained on ImageNet</b> as a feature extractor, then train a new head for
              <b> Normal vs Pneumonia</b>.
            </p>
            <ul className="list">
              <li>Early layers are usually frozen to preserve general features</li>
              <li>The classifier learns medical patterns from X-rays</li>
              <li>Grad-CAM adds explainability by showing influential regions</li>
            </ul>
          </div>
        </div>

      </div>

      <ImageModal
        open={modalOpen}
        title={modalTitle}
        src={modalSrc}
        onClose={() => setModalOpen(false)}
      />
    </div>
  );
}
