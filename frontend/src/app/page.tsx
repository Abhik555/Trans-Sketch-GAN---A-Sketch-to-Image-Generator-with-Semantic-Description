"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Header from "@/components/Header";
import UploadZone from "@/components/UploadZone";
import DrawingCanvas from "@/components/DrawingCanvas";
import ResultPanel from "@/components/ResultPanel";

const API_BASE = "http://localhost:8000";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"upload" | "draw">("upload");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [description, setDescription] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [resultImage, setResultImage] = useState<string>("");
  const [status, setStatus] = useState<{
    type: "info" | "error" | "success";
    icon: string;
    message: string;
  } | null>(null);
  const [isOnline, setIsOnline] = useState(false);
  const [deviceInfo, setDeviceInfo] = useState("");
  const [canvasHasDrawing, setCanvasHasDrawing] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ── Health Check ──
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        const data = await res.json();
        setIsOnline(true);
        setDeviceInfo(data.device?.toUpperCase() || "");
      }
    } catch {
      setIsOnline(false);
      setDeviceInfo("");
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // ── File handling ──
  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setStatus({
        type: "error",
        icon: "⚠️",
        message: "Please upload an image file (PNG, JPG, WEBP).",
      });
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setStatus({
        type: "error",
        icon: "⚠️",
        message: "File too large. Maximum size is 10 MB.",
      });
      return;
    }
    setUploadedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target?.result as string);
    reader.readAsDataURL(file);
    setStatus(null);
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    setPreviewUrl("");
  };

  // ── Generate ──
  const hasSketchInput =
    activeTab === "upload" ? uploadedFile !== null : canvasHasDrawing;
  const canGenerate =
    hasSketchInput && description.trim().length > 0 && !isGenerating;

  const handleGenerate = async () => {
    if (!canGenerate) return;

    setIsGenerating(true);
    setStatus({
      type: "info",
      icon: "⏳",
      message: "Generating image... This may take a few seconds.",
    });

    const formData = new FormData();
    formData.append("description", description.trim());

    try {
      if (activeTab === "upload" && uploadedFile) {
        formData.append("sketch", uploadedFile);
      } else if (activeTab === "draw" && canvasRef.current) {
        const canvasData = canvasRef.current.toDataURL("image/png");
        formData.append("sketch_base64", canvasData);
      } else {
        throw new Error("No sketch input provided.");
      }

      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res
          .json()
          .catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `Server error (${res.status})`);
      }

      const data = await res.json();

      if (data.success && data.image) {
        setResultImage(data.image);
        setStatus({
          type: "success",
          icon: "✅",
          message: "Image generated successfully!",
        });
      } else {
        throw new Error("No image data in response.");
      }
    } catch (err) {
      console.error("Generation error:", err);
      setStatus({
        type: "error",
        icon: "❌",
        message:
          err instanceof Error
            ? err.message
            : "Failed to generate image. Is the backend running?",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <>
      <Header isOnline={isOnline} deviceInfo={deviceInfo} />

      <section className="hero">
        <h1>
          Sketch to <span className="gradient-text">Photorealistic</span> Image
        </h1>
        <p>
          Draw or upload a sketch, describe what you see, and let our GAN
          transform it into a photorealistic image in seconds.
        </p>
      </section>

      <main className="workspace" id="workspace">
        {/* ──── INPUT CARD ──── */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <span>✏️</span> Input Sketch
            </div>
            <div className="tab-group">
              <button
                className={`tab-btn ${activeTab === "upload" ? "active" : ""}`}
                onClick={() => setActiveTab("upload")}
                id="tab-upload-btn"
              >
                📁 Upload
              </button>
              <button
                className={`tab-btn ${activeTab === "draw" ? "active" : ""}`}
                onClick={() => setActiveTab("draw")}
                id="tab-draw-btn"
              >
                🖊️ Draw
              </button>
            </div>
          </div>

          <div className="card-body">
            {/* Upload Tab */}
            <div
              className={`tab-content ${activeTab === "upload" ? "active" : ""}`}
            >
              <UploadZone
                previewUrl={previewUrl}
                onFileSelect={handleFileSelect}
                onRemove={handleRemoveFile}
              />
            </div>

            {/* Draw Tab */}
            <div
              className={`tab-content ${activeTab === "draw" ? "active" : ""}`}
            >
              <DrawingCanvas
                canvasRef={canvasRef}
                onDrawingChange={setCanvasHasDrawing}
              />
            </div>

            {/* Description */}
            <div className="description-section">
              <label className="input-label" htmlFor="description-input">
                📝 Text Description
                <span className="char-count">{description.length} / 200</span>
              </label>
              <textarea
                className="description-input"
                id="description-input"
                placeholder="e.g., A young woman with wavy brown hair, green eyes, wearing a red dress..."
                maxLength={200}
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>

            {/* Generate Button */}
            <button
              className={`generate-btn ${isGenerating ? "loading" : ""}`}
              id="generate-btn"
              disabled={!canGenerate}
              onClick={handleGenerate}
            >
              {isGenerating ? (
                <div className="spinner" />
              ) : (
                <>
                  <span className="btn-icon">⚡</span>
                  <span className="btn-text">Generate Image</span>
                </>
              )}
            </button>

            {/* Status */}
            {status && (
              <div className={`status-bar ${status.type}`}>
                <span>{status.icon}</span>
                <span>{status.message}</span>
              </div>
            )}
          </div>
        </div>

        {/* ──── OUTPUT CARD ──── */}
        <ResultPanel
          resultImage={resultImage}
          onNewGeneration={() => {
            setResultImage("");
            setStatus(null);
          }}
        />
      </main>

      <footer className="footer">
        <p>SketchGAN — Sketch-to-Image Generative Adversarial Network</p>
      </footer>
    </>
  );
}
