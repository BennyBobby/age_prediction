import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://localhost:8000/predict",
        formData,
      );
      setPrediction(response.data.predicted_age);
    } catch (e) {
      console.error("Error during prediction:", e);
    }
  };

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h1>Age Prediction AI</h1>

      <input type="file" onChange={(e) => setFile(e.target.files[0])} />

      <button onClick={handleUpload} style={{ marginLeft: "10px" }}>
        Analyze Image
      </button>

      {prediction && <h2>Estimated Age: {prediction} years old</h2>}
    </div>
  );
}

export default App;
