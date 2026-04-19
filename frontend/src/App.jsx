import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPrediction(null);
    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setPrediction(null);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await axios.post(`${apiUrl}/predict`, formData);
      setPrediction(response.data.predicted_age);
    } catch (e) {
      console.error("Error during prediction:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h1>Age Prediction AI</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      <button onClick={handleUpload} style={{ marginLeft: "10px" }}>
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

      {prediction !== null && !loading && (
        <h2 style={{ marginTop: "20px" }}>Estimated Age: {prediction} years old</h2>
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
