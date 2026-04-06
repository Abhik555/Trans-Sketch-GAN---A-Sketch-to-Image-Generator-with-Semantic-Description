"use client";

import { useRef, useState, DragEvent } from "react";

interface UploadZoneProps {
  previewUrl: string;
  onFileSelect: (file: File) => void;
  onRemove: () => void;
}

export default function UploadZone({
  previewUrl,
  onFileSelect,
  onRemove,
}: UploadZoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleClick = () => fileInputRef.current?.click();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  if (previewUrl) {
    return (
      <div className="upload-preview">
        <img src={previewUrl} alt="Uploaded sketch preview" />
        <button className="remove-btn" onClick={onRemove} title="Remove image">
          ✕
        </button>
      </div>
    );
  }

  return (
    <>
      <div
        className={`upload-zone ${isDragOver ? "drag-over" : ""}`}
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        id="upload-zone"
      >
        <span className="upload-icon">📤</span>
        <p className="upload-text">
          <strong>Click to upload</strong> or drag &amp; drop
        </p>
        <p className="upload-hint">PNG, JPG, or WEBP • Max 10 MB</p>
      </div>
      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        hidden
        onChange={handleChange}
        id="file-input"
      />
    </>
  );
}
