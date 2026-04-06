"use client";

import { useEffect, useRef, useState, useCallback, RefObject } from "react";

interface DrawingCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  onDrawingChange: (hasDrawing: boolean) => void;
}

export default function DrawingCanvas({
  canvasRef,
  onDrawingChange,
}: DrawingCanvasProps) {
  const [currentTool, setCurrentTool] = useState<"pen" | "eraser">("pen");
  const [brushSize, setBrushSize] = useState(4);
  const isDrawing = useRef(false);
  const strokeHistory = useRef<ImageData[]>([]);

  // ── Initialize canvas ──
  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    strokeHistory.current = [
      ctx.getImageData(0, 0, canvas.width, canvas.height),
    ];
    onDrawingChange(false);
  }, [canvasRef, onDrawingChange]);

  useEffect(() => {
    initCanvas();
  }, [initCanvas]);

  const saveState = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    strokeHistory.current.push(
      ctx.getImageData(0, 0, canvas.width, canvas.height)
    );
    if (strokeHistory.current.length > 50) strokeHistory.current.shift();
  };

  const checkIfDrawn = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const step = Math.floor(data.length / 400) * 4;
    for (let i = 0; i < data.length; i += step) {
      if (data[i] < 250 || data[i + 1] < 250 || data[i + 2] < 250) {
        onDrawingChange(true);
        return;
      }
    }
    onDrawingChange(false);
  }, [canvasRef, onDrawingChange]);

  const getPos = (
    e: React.MouseEvent | React.TouchEvent | MouseEvent | TouchEvent
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX: number, clientY: number;
    if ("touches" in e && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else if ("clientX" in e) {
      clientX = e.clientX;
      clientY = e.clientY;
    } else {
      return { x: 0, y: 0 };
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    isDrawing.current = true;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing.current) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pos = getPos(e);
    const scale = canvas.width / canvas.getBoundingClientRect().width;
    ctx.lineWidth = brushSize * scale;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = currentTool === "pen" ? "#000000" : "#ffffff";

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  const endDraw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing.current) return;
    e.preventDefault();
    isDrawing.current = false;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.closePath();
    saveState();
    checkIfDrawn();
  };

  const handleUndo = () => {
    if (strokeHistory.current.length > 1) {
      strokeHistory.current.pop();
      const prev = strokeHistory.current[strokeHistory.current.length - 1];
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.putImageData(prev, 0, 0);
      checkIfDrawn();
    }
  };

  const handleClear = () => {
    initCanvas();
  };

  return (
    <div className="canvas-container">
      <canvas
        className="drawing-canvas"
        ref={canvasRef}
        width={512}
        height={512}
        id="drawing-canvas"
        style={{ cursor: currentTool === "pen" ? "crosshair" : "cell" }}
        onMouseDown={startDraw}
        onMouseMove={draw}
        onMouseUp={endDraw}
        onMouseLeave={endDraw}
        onTouchStart={startDraw}
        onTouchMove={draw}
        onTouchEnd={endDraw}
      />
      <div className="canvas-toolbar">
        <button
          className={`tool-btn ${currentTool === "pen" ? "active" : ""}`}
          onClick={() => setCurrentTool("pen")}
          title="Pen"
          id="tool-pen"
        >
          ✏️
        </button>
        <button
          className={`tool-btn ${currentTool === "eraser" ? "active" : ""}`}
          onClick={() => setCurrentTool("eraser")}
          title="Eraser"
          id="tool-eraser"
        >
          🧹
        </button>
        <div className="tool-separator" />
        <button
          className="tool-btn"
          onClick={handleUndo}
          title="Undo"
          id="tool-undo"
        >
          ↩️
        </button>
        <button
          className="tool-btn"
          onClick={handleClear}
          title="Clear Canvas"
          id="tool-clear"
        >
          🗑️
        </button>
        <div className="tool-separator" />
        <input
          type="range"
          className="brush-size-slider"
          min={1}
          max={30}
          value={brushSize}
          onChange={(e) => setBrushSize(parseInt(e.target.value))}
          id="brush-size"
        />
        <span className="brush-size-label">{brushSize}px</span>
      </div>
    </div>
  );
}
