"use client";

interface ResultPanelProps {
  resultImage: string;
  onNewGeneration: () => void;
}

export default function ResultPanel({
  resultImage,
  onNewGeneration,
}: ResultPanelProps) {
  return (
    <div className="card output-card">
      <div className="card-header">
        <div className="card-title">
          <span>🖼️</span> Generated Result
        </div>
      </div>
      <div className="card-body">
        {!resultImage ? (
          <div className="result-placeholder" id="result-placeholder">
            <span className="placeholder-icon">🌌</span>
            <p className="placeholder-text">
              Your generated image will appear here.
              <br />
              Draw or upload a sketch to begin.
            </p>
          </div>
        ) : (
          <div className="result-image-container" id="result-container">
            <img
              className="result-image"
              src={resultImage}
              alt="AI generated image"
              id="result-image"
            />
            <div className="result-actions">
              <a
                className="action-btn primary"
                href={resultImage}
                download={`sketchgan_${Date.now()}.png`}
                id="download-btn"
              >
                💾 Download
              </a>
              <button
                className="action-btn"
                onClick={onNewGeneration}
                id="new-gen-btn"
              >
                🔄 New Generation
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
