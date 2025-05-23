import React, { useRef, useState } from "react";
import { ReactSketchCanvas } from "react-sketch-canvas";
import "./App.css";

function App() {
  const canvasRef = useRef();
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async () => {
    const base64 = await canvasRef.current?.exportImage("png");

    // Convert base64 to Blob
    const blob = dataURLtoBlob(base64);

    const formData = new FormData();
    formData.append("image", blob, "digit.png");

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    setPrediction(result.predictions);
  };

  const dataURLtoBlob = (dataURL) => {
    const byteString = atob(dataURL.split(",")[1]);
    const mimeString = dataURL.split(",")[0].split(":")[1].split(";")[0];

    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ab], { type: mimeString });
  };

  return (
    <div className="App">
      <h1>DigitNet</h1>
      <p className="instructions">
        Draw a digit (0–9) in the canvas below, then click <strong>Predict</strong> to see what the model thinks.
        You can clear and try again as many times as you like!
      </p>

      <div className="canvas-container">
        <ReactSketchCanvas
          ref={canvasRef}
          width="280px"
          height="280px"
          strokeWidth={15}
          strokeColor="black"
        />
      </div>

      <button onClick={handleSubmit}>Predict</button>
      <button onClick={() => canvasRef.current?.clearCanvas()}>Clear</button>

      {prediction && (
        <div className="prediction-box">
          <h3>Model Confidence</h3>
          <div className="confidence-dots">
            {prediction.map((p, index) => (
              <div className="dot-container" key={index}>
                <div
                  className="dot"
                  style={{
                    width: `${20 + p.confidence * 30}px`,
                    height: `${20 + p.confidence * 30}px`,
                    opacity: 0.3 + p.confidence * 0.7,
                    backgroundColor: "#3b82f6",
                  }}
                />
                <div className="digit-label">{p.digit}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
