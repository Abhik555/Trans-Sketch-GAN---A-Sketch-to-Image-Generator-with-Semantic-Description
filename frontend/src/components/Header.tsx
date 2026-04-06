"use client";

interface HeaderProps {
  isOnline: boolean;
  deviceInfo: string;
}

export default function Header({ isOnline, deviceInfo }: HeaderProps) {
  return (
    <header className="header" id="app-header">
      <div className="header-inner">
        <div className="logo">
          <div className="logo-icon">🎨</div>
          <span className="logo-text">SketchGAN</span>
        </div>
        <span
          className={`header-badge ${isOnline ? "badge-online" : "badge-offline"}`}
          id="status-badge"
        >
          {isOnline
            ? `● ONLINE${deviceInfo ? ` — ${deviceInfo}` : ""}`
            : "● OFFLINE"}
        </span>
      </div>
    </header>
  );
}
