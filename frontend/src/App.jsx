import React, { useState } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setResult(null);
    setError(null);
    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        timeout: 30000,
      });
      setResult(response.data);
    } catch (e) {
      if (e.code === "ECONNABORTED") {
        setError("Request timed out. The server took too long to respond.");
      } else if (e.response) {
        setError(e.response.data?.detail || "Prediction failed.");
      } else {
        setError("Cannot reach the server. Make sure the API is running.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h1>Age Prediction AI</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      <button onClick={handleUpload} style={{ marginLeft: "10px" }} disabled={loading}>
        Analyze Image
      </button>

      {preview && (
        <div style={{ marginTop: "20px" }}>
          <img
            src={preview}
            alt="Preview"
            style={{ maxWidth: "300px", maxHeight: "300px", borderRadius: "8px", objectFit: "cover" }}
          />
        </div>
      )}

      {loading && (
        <div style={{ marginTop: "20px" }}>
          <div style={{
            width: "200px",
            height: "8px",
            background: "#e0e0e0",
            borderRadius: "4px",
            margin: "0 auto",
            overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              background: "#4f46e5",
              borderRadius: "4px",
              animation: "loading 1.5s infinite ease-in-out",
              width: "40%",
            }} />
          </div>
          <p style={{ marginTop: "8px", color: "#666" }}>Analyzing...</p>
        </div>
      )}

      {error && (
        <p style={{ marginTop: "20px", color: "#dc2626", fontWeight: 500 }}>
          {error}
        </p>
      )}

      {result && !loading && (
        <div style={{ marginTop: "20px" }}>
          {result.face_crop && (
            <div style={{ marginBottom: "12px" }}>
              <p style={{ color: "#6b7280", fontSize: "0.85rem", marginBottom: "6px" }}>Detected face</p>
              <img
                src={result.face_crop}
                alt="Detected face"
                style={{ width: "120px", height: "120px", objectFit: "cover", borderRadius: "8px", border: "2px solid #4f46e5" }}
              />
            </div>
          )}
          <h2>Estimated Age: {result.predicted_age} years old</h2>
          <p style={{ color: "#6b7280", fontSize: "0.95rem" }}>
            Range: {result.confidence_interval.low} – {result.confidence_interval.high} years (95% confidence)
          </p>
        </div>
      )}

      <style>{`
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(350%); }
        }
      `}</style>
    </div>
  );
}

export default App;
