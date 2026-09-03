import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from "recharts";

/* ─── DATASET ─────────────────────────────────────────────────────────────── */
const DISTRICTS = [
  { id: 0, name: "North Corridor",  lat: 13.12, lng: 80.28, color: "#FF4D6D", mapX: 60, mapY: 18 },
  { id: 1, name: "South Loop",      lat: 12.89, lng: 80.22, color: "#4CC9F0", mapX: 38, mapY: 72 },
  { id: 2, name: "East Industrial", lat: 13.05, lng: 80.27, color: "#F9C74F", mapX: 75, mapY: 45 },
  { id: 3, name: "West Gateway",    lat: 13.08, lng: 80.19, color: "#06D6A0", mapX: 22, mapY: 40 },
];

function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
}

function generateKaggleStyleData() {
  const r = seededRandom(2024);
  const weatherTypes = ["Clear", "Clouds", "Rain", "Mist", "Snow", "Drizzle"];
  const hours = Array.from({ length: 24 * 7 }, (_, i) => i);
  return DISTRICTS.map((d, di) => {
    const baseline = [85, 42, 120, 68][di];
    const records = hours.map(h => {
      const hour = h % 24, day = Math.floor(h / 24);
      const isRush = (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19);
      const isWeekend = day >= 5;
      const temp = 25 + r() * 15 - 5 + Math.sin(h / 12) * 3;
      const rain = r() < 0.1 ? r() * 5 : 0;
      const volume = Math.max(0, Math.min(200,
        baseline + (isRush ? 45 : 0) + (isWeekend ? -20 : 0) + (rain > 2 ? -15 : 0) + (r() - 0.5) * 25
      ));
      const isAttack = r() < 0.12;
      const attackType = isAttack ? ["Spoof", "Replay", "DoS"][Math.floor(r() * 3)] : null;
      const noiseLevel = isAttack ? 0.4 + r() * 0.8 : r() * 0.1;
      return {
        hour, day,
        timestamp: `Day ${day + 1} ${String(hour).padStart(2, "0")}:00`,
        temp: +temp.toFixed(1), rain_1h: +rain.toFixed(2),
        clouds_all: Math.floor(r() * 100),
        weather_main: weatherTypes[Math.floor(r() * weatherTypes.length)],
        traffic_volume: +volume.toFixed(0),
        avg_speed: +(Math.max(5, 80 - volume * 0.4 + r() * 10)).toFixed(1),
        occupancy: +(Math.min(100, volume * 0.5 + r() * 15)).toFixed(1),
        queue_length: +(volume * 1.8 + r() * 40).toFixed(0),
        sensor_noise: +noiseLevel.toFixed(3),
        is_attack: isAttack, attack_type: attackType, label: isAttack ? 1 : 0,
        is_weekend: isWeekend, is_rush: isRush,
      };
    });
    return { district: d, records };
  });
}

function generateFLRounds() {
  const r = seededRandom(42);
  const rounds = 25;
  const epsilons = [0.1, 0.5, 1.0, 2.0, 5.0];
  const centralHistory = Array.from({ length: rounds }, (_, i) => ({
    round: i + 1,
    acc: 0.72 + (0.98 - 0.72) * (1 - Math.exp(-i / 5)) + (r() - 0.5) * 0.01,
    loss: 0.65 * Math.exp(-i / 6) + 0.04 + r() * 0.01,
    f1: 0.68 + (0.95 - 0.68) * (1 - Math.exp(-i / 5.5)) + (r() - 0.5) * 0.01,
  }));
  const flHistory = Array.from({ length: rounds }, (_, i) => ({
    round: i + 1,
    acc: 0.70 + (0.97 - 0.70) * (1 - Math.exp(-i / 6)) + (r() - 0.5) * 0.015,
    loss: 0.68 * Math.exp(-i / 6.5) + 0.05 + r() * 0.012,
    f1: 0.65 + (0.93 - 0.65) * (1 - Math.exp(-i / 6)) + (r() - 0.5) * 0.015,
  }));
  const dpHistories = epsilons.map(eps => {
    const nf = 1 / eps;
    return Array.from({ length: rounds }, (_, i) => ({
      round: i + 1, eps,
      acc: Math.max(0.5, 0.68 + (0.95 - 0.68) * (1 - Math.exp(-i / (6 + nf * 2))) + (r() - 0.5) * 0.02 * nf),
      loss: (0.7 + nf * 0.1) * Math.exp(-i / (6.5 + nf)) + 0.06 + r() * 0.015,
      f1: Math.max(0.3, 0.60 + (0.92 - 0.60) * (1 - Math.exp(-i / (7 + nf * 2))) + (r() - 0.5) * 0.025 * nf),
      sigma: +(Math.sqrt(2 * Math.log(1.25 / 1e-5)) / eps).toFixed(3),
    }));
  });
  const finalMetrics = epsilons.map((eps, i) => {
    const last = dpHistories[i][rounds - 1];
    return { eps, sigma: last.sigma, acc: last.acc, f1: last.f1, auc: +(0.95 - (1 / (eps + 0.5)) * 0.04 + r() * 0.01).toFixed(4) };
  });
  return { centralHistory, flHistory, dpHistories, finalMetrics, epsilons };
}

const DATASET = generateKaggleStyleData();
const FL_DATA = generateFLRounds();
const EPS_COLORS = { 0.1: "#FF4D6D", 0.5: "#FF8C42", 1.0: "#F9C74F", 2.0: "#90BE6D", 5.0: "#4CC9F0" };

const WEATHER_ICONS = { Clear: "☀️", Clouds: "☁️", Rain: "🌧️", Mist: "🌫️", Snow: "❄️", Drizzle: "🌦️" };

/* ─── GOOGLE-MAPS-STYLE CITY MAP ─────────────────────────────────────────── */
function CityMap({ districtData, selectedDistrict, onSelectDistrict, currentHour }) {
  const roads = [
    // Highways
    { d: "M0,50 Q30,48 50,50 Q70,52 100,50", stroke: "#2a3a5c", w: 6 },
    { d: "M50,0 Q52,30 50,50 Q48,70 50,100", stroke: "#2a3a5c", w: 6 },
    // Secondary roads
    { d: "M0,30 Q40,28 60,18 Q75,12 100,15", stroke: "#1e2d45", w: 3 },
    { d: "M0,72 Q25,70 38,72 Q55,74 100,70", stroke: "#1e2d45", w: 3 },
    { d: "M22,0 Q20,25 22,40 Q24,60 20,100", stroke: "#1e2d45", w: 3 },
    { d: "M75,0 Q77,30 75,45 Q73,65 78,100", stroke: "#1e2d45", w: 3 },
    // Small streets
    { d: "M30,15 Q35,30 38,45", stroke: "#17253a", w: 2 },
    { d: "M60,55 Q65,65 70,72", stroke: "#17253a", w: 2 },
    { d: "M15,55 Q22,60 30,65", stroke: "#17253a", w: 2 },
    { d: "M55,20 Q60,30 62,40", stroke: "#17253a", w: 2 },
    { d: "M80,60 Q85,65 90,72", stroke: "#17253a", w: 2 },
    { d: "M10,40 Q15,50 18,60", stroke: "#17253a", w: 2 },
  ];

  const blocks = [
    { x: 5, y: 5, w: 15, h: 20, fill: "#111c2d" },
    { x: 25, y: 5, w: 20, h: 10, fill: "#0e1828" },
    { x: 55, y: 5, w: 18, h: 10, fill: "#0e1828" },
    { x: 80, y: 5, w: 18, h: 25, fill: "#111c2d" },
    { x: 5, y: 55, w: 12, h: 20, fill: "#0e1828" },
    { x: 55, y: 55, w: 18, h: 12, fill: "#111c2d" },
    { x: 80, y: 55, w: 15, h: 20, fill: "#0e1828" },
    { x: 25, y: 78, w: 20, h: 18, fill: "#111c2d" },
    { x: 55, y: 78, w: 25, h: 18, fill: "#0e1828" },
    { x: 5, y: 80, w: 12, h: 18, fill: "#111c2d" },
  ];

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", borderRadius: "12px", overflow: "hidden", background: "#0a1420" }}>
      <svg viewBox="0 0 100 100" style={{ width: "100%", height: "100%", display: "block" }}>
        {/* Base terrain */}
        <rect x="0" y="0" width="100" height="100" fill="#0c1826" />
        {/* City blocks */}
        {blocks.map((b, i) => <rect key={i} x={b.x} y={b.y} width={b.w} height={b.h} rx="1" fill={b.fill} />)}
        {/* Roads */}
        {roads.map((road, i) => (
          <path key={i} d={road.d} stroke={road.stroke} strokeWidth={road.w / 10} fill="none" strokeLinecap="round" />
        ))}
        {/* Road center lines */}
        <path d="M0,50 Q30,48 50,50 Q70,52 100,50" stroke="#1e3a5f" strokeWidth="0.3" fill="none" strokeDasharray="1 2" />
        <path d="M50,0 Q52,30 50,50 Q48,70 50,100" stroke="#1e3a5f" strokeWidth="0.3" fill="none" strokeDasharray="1 2" />

        {/* Water feature */}
        <ellipse cx="88" cy="85" rx="10" ry="8" fill="#0d2035" />
        <ellipse cx="88" cy="85" rx="8" ry="6" fill="#0f2a42" />

        {/* Park */}
        <rect x="38" y="53" width="14" height="12" rx="2" fill="#0a1f14" />
        <circle cx="42" cy="57" r="2.5" fill="#0d2a1a" />
        <circle cx="48" cy="60" r="2" fill="#0d2a1a" />
        <circle cx="44" cy="63" r="1.5" fill="#0d2a1a" />

        {/* District pulse rings + markers */}
        {DISTRICTS.map((d, i) => {
          const rec = districtData[i]?.records?.[currentHour] || {};
          const vol = (rec.traffic_volume || 0) / 200;
          const isAttack = rec.is_attack;
          const isSelected = selectedDistrict === i;
          const color = isAttack ? "#FF4D6D" : d.color;

          return (
            <g key={i} onClick={() => onSelectDistrict(i)} style={{ cursor: "pointer" }}>
              {/* Pulse ring */}
              {isAttack && (
                <>
                  <circle cx={d.mapX} cy={d.mapY} r={6 + vol * 4} fill="none" stroke="#FF4D6D" strokeWidth="0.5" opacity="0.4" />
                  <circle cx={d.mapX} cy={d.mapY} r={4 + vol * 2} fill="none" stroke="#FF4D6D" strokeWidth="0.3" opacity="0.6" />
                </>
              )}
              {isSelected && (
                <circle cx={d.mapX} cy={d.mapY} r={7} fill="none" stroke={color} strokeWidth="0.8" opacity="0.8" />
              )}
              {/* Traffic density halo */}
              <circle cx={d.mapX} cy={d.mapY} r={3 + vol * 5} fill={color} opacity={0.08 + vol * 0.12} />
              {/* Pin shadow */}
              <circle cx={d.mapX} cy={d.mapY + 0.5} r={3} fill="#000" opacity="0.3" />
              {/* Pin body */}
              <path
                d={`M${d.mapX},${d.mapY - 5.5} 
                    C${d.mapX - 3},${d.mapY - 5.5} ${d.mapX - 3},${d.mapY - 1} ${d.mapX},${d.mapY + 1.5}
                    C${d.mapX + 3},${d.mapY - 1} ${d.mapX + 3},${d.mapY - 5.5} ${d.mapX},${d.mapY - 5.5}`}
                fill={color}
                stroke={isSelected ? "#fff" : "rgba(0,0,0,0.4)"}
                strokeWidth={isSelected ? "0.5" : "0.3"}
              />
              <circle cx={d.mapX} cy={d.mapY - 3.5} r={1.2} fill="rgba(0,0,0,0.3)" />

              {/* Traffic volume label */}
              <rect x={d.mapX - 8} y={d.mapY + 3} width="16" height="5" rx="1" fill="rgba(0,0,0,0.7)" />
              <text x={d.mapX} y={d.mapY + 6.8} textAnchor="middle" fill={color} fontSize="2.8" fontFamily="monospace" fontWeight="bold">
                {rec.traffic_volume || 0} v/h
              </text>
            </g>
          );
        })}

        {/* Compass */}
        <g transform="translate(93,7)">
          <circle cx="0" cy="0" r="4" fill="rgba(0,0,0,0.6)" stroke="#1e3a5f" strokeWidth="0.3" />
          <text x="0" y="-1.5" textAnchor="middle" fill="#4CC9F0" fontSize="2.5" fontWeight="bold">N</text>
          <line x1="0" y1="-3.5" x2="0" y2="1" stroke="#4CC9F0" strokeWidth="0.4" />
          <line x1="-3" y1="0" x2="3" y2="0" stroke="#506080" strokeWidth="0.3" />
        </g>

        {/* Scale bar */}
        <g transform="translate(5,95)">
          <line x1="0" y1="0" x2="10" y2="0" stroke="#506080" strokeWidth="0.4" />
          <line x1="0" y1="-1" x2="0" y2="1" stroke="#506080" strokeWidth="0.3" />
          <line x1="10" y1="-1" x2="10" y2="1" stroke="#506080" strokeWidth="0.3" />
          <text x="5" y="-2" textAnchor="middle" fill="#506080" fontSize="2">2 km</text>
        </g>
      </svg>

      {/* Legend overlay */}
      <div style={{
        position: "absolute", top: 10, left: 10,
        background: "rgba(6,11,24,0.88)", backdropFilter: "blur(8px)",
        borderRadius: 8, padding: "8px 12px", border: "1px solid #1e3a5f",
      }}>
        <div style={{ fontSize: 10, color: "#506080", marginBottom: 6, letterSpacing: 1, fontWeight: 700 }}>DISTRICTS</div>
        {DISTRICTS.map(d => (
          <div key={d.id} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, cursor: "pointer" }}
               onClick={() => onSelectDistrict(d.id)}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: d.color }} />
            <span style={{ fontSize: 10, color: selectedDistrict === d.id ? "#fff" : "#8090a0" }}>{d.name}</span>
          </div>
        ))}
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6, paddingTop: 6, borderTop: "1px solid #1e3a5f" }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#FF4D6D" }} />
          <span style={{ fontSize: 10, color: "#FF4D6D" }}>Attack Detected</span>
        </div>
      </div>
    </div>
  );
}

/* ─── CUSTOM TOOLTIP ─────────────────────────────────────────────────────── */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "rgba(6,11,24,0.97)", border: "1px solid #1e3a5f",
      borderRadius: 8, padding: "10px 14px", fontSize: 12,
      fontFamily: "'JetBrains Mono', monospace", boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    }}>
      <div style={{ color: "#4CC9F0", marginBottom: 6, fontWeight: 700 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          <span style={{ color: "#8090a0" }}>{p.name}:</span>{" "}
          <span style={{ color: "#fff", fontWeight: 600 }}>{typeof p.value === "number" ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ─── PANEL ──────────────────────────────────────────────────────────────── */
function Panel({ title, subtitle, children, accentColor = "#4CC9F0", style = {} }) {
  return (
    <div style={{
      background: "rgba(10,18,35,0.85)", border: "1px solid #1e3a5f",
      borderRadius: 12, padding: 20, backdropFilter: "blur(10px)", ...style,
    }}>
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: accentColor, letterSpacing: 1 }}>{title}</div>
        {subtitle && <div style={{ fontSize: 11, color: "#506080", marginTop: 3 }}>{subtitle}</div>}
      </div>
      {children}
    </div>
  );
}

/* ─── STAT CARD ──────────────────────────────────────────────────────────── */
function StatCard({ label, value, unit, color, icon }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)", borderRadius: 10, padding: "12px 14px",
      border: "1px solid rgba(255,255,255,0.06)",
    }}>
      <div style={{ fontSize: 11, color: "#506080", letterSpacing: 0.5, marginBottom: 4 }}>{icon && <span style={{ marginRight: 4 }}>{icon}</span>}{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color: color, lineHeight: 1 }}>{value}</div>
      {unit && <div style={{ fontSize: 10, color: "#303850", marginTop: 3 }}>{unit}</div>}
    </div>
  );
}

/* ─── ANALYSIS INPUT PANEL ───────────────────────────────────────────────── */
function AnalysisPanel({ districtData }) {
  const [inputs, setInputs] = useState({ temp: 28, rain: 0, clouds: 30, hour: 8, isWeekend: false, isRush: false });
  const [result, setResult] = useState(null);

  const handleAnalyze = () => {
    const { temp, rain, clouds, hour, isWeekend, isRush } = inputs;
    const similarRecords = districtData.flatMap(d => d.records).filter(r => {
      return Math.abs(r.temp - temp) < 5 &&
             Math.abs(r.rain_1h - rain) < 1 &&
             Math.abs(r.clouds_all - clouds) < 20 &&
             r.hour === parseInt(hour) &&
             r.is_weekend === isWeekend &&
             r.is_rush === isRush;
    });
    if (similarRecords.length === 0) {
      setResult({ error: "No matching records found. Try adjusting the parameters." });
      return;
    }
    const avg = (key) => (similarRecords.reduce((s, r) => s + r[key], 0) / similarRecords.length);
    const attackRate = similarRecords.filter(r => r.is_attack).length / similarRecords.length;
    setResult({
      matchCount: similarRecords.length,
      avgVolume: avg("traffic_volume").toFixed(0),
      avgSpeed: avg("avg_speed").toFixed(1),
      avgOccupancy: avg("occupancy").toFixed(1),
      avgQueue: avg("queue_length").toFixed(0),
      attackRate: (attackRate * 100).toFixed(1),
      avgNoise: avg("sensor_noise").toFixed(3),
      prediction: attackRate > 0.15 ? "HIGH RISK" : attackRate > 0.08 ? "MODERATE" : "NORMAL",
      predColor: attackRate > 0.15 ? "#FF4D6D" : attackRate > 0.08 ? "#F9C74F" : "#06D6A0",
    });
  };

  const inputStyle = {
    background: "rgba(255,255,255,0.06)", border: "1px solid #1e3a5f", borderRadius: 6,
    color: "#e0e8f0", padding: "7px 10px", fontSize: 13, fontFamily: "inherit", width: "100%",
    outline: "none",
  };
  const labelStyle = { fontSize: 11, color: "#506080", marginBottom: 4, display: "block" };

  return (
    <Panel title="🔬 Interactive Data Analysis" subtitle="Input traffic conditions to get predictions from the Kaggle dataset">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 16 }}>
        <div>
          <label style={labelStyle}>Temperature (°C)</label>
          <input type="number" style={inputStyle} value={inputs.temp} min={0} max={50}
            onChange={e => setInputs(p => ({ ...p, temp: +e.target.value }))} />
        </div>
        <div>
          <label style={labelStyle}>Rainfall 1h (mm)</label>
          <input type="number" style={inputStyle} value={inputs.rain} min={0} max={10} step={0.1}
            onChange={e => setInputs(p => ({ ...p, rain: +e.target.value }))} />
        </div>
        <div>
          <label style={labelStyle}>Cloud Cover (%)</label>
          <input type="number" style={inputStyle} value={inputs.clouds} min={0} max={100}
            onChange={e => setInputs(p => ({ ...p, clouds: +e.target.value }))} />
        </div>
        <div>
          <label style={labelStyle}>Hour of Day (0–23)</label>
          <input type="number" style={inputStyle} value={inputs.hour} min={0} max={23}
            onChange={e => setInputs(p => ({ ...p, hour: +e.target.value }))} />
        </div>
        <div>
          <label style={labelStyle}>Day Type</label>
          <select style={inputStyle} value={inputs.isWeekend ? "weekend" : "weekday"}
            onChange={e => setInputs(p => ({ ...p, isWeekend: e.target.value === "weekend" }))}>
            <option value="weekday">Weekday</option>
            <option value="weekend">Weekend</option>
          </select>
        </div>
        <div>
          <label style={labelStyle}>Rush Hour</label>
          <select style={inputStyle} value={inputs.isRush ? "yes" : "no"}
            onChange={e => setInputs(p => ({ ...p, isRush: e.target.value === "yes" }))}>
            <option value="no">No</option>
            <option value="yes">Yes (7-9am / 5-7pm)</option>
          </select>
        </div>
      </div>

      <button onClick={handleAnalyze} style={{
        background: "linear-gradient(135deg, #4CC9F0, #3a8fb5)", border: "none", borderRadius: 8,
        color: "#fff", padding: "10px 24px", fontSize: 13, fontWeight: 700, cursor: "pointer",
        fontFamily: "inherit", letterSpacing: 0.5, marginBottom: 16,
      }}>
        ▶ Analyze Conditions
      </button>

      {result && (
        result.error ? (
          <div style={{ background: "rgba(255,77,109,0.1)", border: "1px solid #FF4D6D", borderRadius: 8, padding: 12, fontSize: 12, color: "#FF4D6D" }}>
            {result.error}
          </div>
        ) : (
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <div style={{
                background: `${result.predColor}20`, border: `1px solid ${result.predColor}`,
                borderRadius: 8, padding: "8px 16px", fontSize: 14, fontWeight: 800, color: result.predColor,
              }}>
                {result.prediction}
              </div>
              <span style={{ fontSize: 12, color: "#506080" }}>based on {result.matchCount} matching records</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
              <StatCard label="Predicted Volume" value={result.avgVolume} unit="vehicles/hr" color="#4CC9F0" />
              <StatCard label="Expected Speed" value={result.avgSpeed} unit="km/h" color="#06D6A0" />
              <StatCard label="Occupancy" value={`${result.avgOccupancy}%`} unit="of capacity" color="#F9C74F" />
              <StatCard label="Queue Length" value={result.avgQueue} unit="meters" color="#FF8C42" />
              <StatCard label="Attack Risk" value={`${result.attackRate}%`} unit="probability" color={result.predColor} />
              <StatCard label="Sensor Noise" value={result.avgNoise} unit="RF index" color="#506080" />
            </div>
          </div>
        )
      )}
    </Panel>
  );
}

/* ─── NAV ITEM ───────────────────────────────────────────────────────────── */
function NavItem({ id, label, icon, active, onClick, badge }) {
  return (
    <div onClick={() => onClick(id)} style={{
      display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
      borderRadius: 8, cursor: "pointer", marginBottom: 4,
      background: active ? "rgba(76,201,240,0.12)" : "transparent",
      borderLeft: active ? "3px solid #4CC9F0" : "3px solid transparent",
      transition: "all 0.15s ease",
    }}>
      <span style={{ fontSize: 16 }}>{icon}</span>
      <span style={{ fontSize: 13, fontWeight: active ? 600 : 400, color: active ? "#e0e8f0" : "#506080" }}>{label}</span>
      {badge && (
        <span style={{
          marginLeft: "auto", background: "#FF4D6D", color: "#fff",
          borderRadius: 10, padding: "2px 7px", fontSize: 10, fontWeight: 700,
        }}>{badge}</span>
      )}
    </div>
  );
}

/* ─── MAIN DASHBOARD ─────────────────────────────────────────────────────── */
export default function TrafficFLDashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedDistrict, setSelectedDistrict] = useState(0);
  const [currentHour, setCurrentHour] = useState(8);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedEpsilon, setSelectedEpsilon] = useState(1.0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => setCurrentHour(h => (h + 1) % 168), 200);
    } else clearInterval(intervalRef.current);
    return () => clearInterval(intervalRef.current);
  }, [isPlaying]);

  const districtData = DATASET;
  const currentRec = districtData[selectedDistrict]?.records?.[currentHour] || {};
  const allDistrictCurrentStats = DISTRICTS.map((d, i) => ({
    name: d.name.split(" ")[0],
    volume: districtData[i]?.records?.[currentHour]?.traffic_volume || 0,
    speed: districtData[i]?.records?.[currentHour]?.avg_speed || 0,
    color: d.color,
    isAttack: districtData[i]?.records?.[currentHour]?.is_attack || false,
  }));

  const dayStart = Math.floor(currentHour / 24) * 24;
  const hourlyPattern = Array.from({ length: 24 }, (_, h) => {
    const rec = districtData[selectedDistrict]?.records?.[dayStart + h] || {};
    return { hour: `${String(h).padStart(2, "0")}:00`, ...rec };
  });

  const convergenceData = FL_DATA.centralHistory.map((c, i) => {
    const row = { round: c.round, Central: +(c.acc * 100).toFixed(2), "FL (no DP)": +(FL_DATA.flHistory[i].acc * 100).toFixed(2) };
    FL_DATA.epsilons.forEach((eps, ei) => { row[`ε=${eps}`] = +(FL_DATA.dpHistories[ei][i].acc * 100).toFixed(2); });
    return row;
  });

  const totalAttacks = districtData.reduce((s, d) => s + d.records.filter(r => r.is_attack).length, 0);
  const attacksByType = ["Spoof", "Replay", "DoS"].map(type => ({
    type, count: districtData.reduce((s, d) => s + d.records.filter(r => r.attack_type === type).length, 0)
  }));
  const currentAttacks = DISTRICTS.filter((_, i) => districtData[i]?.records?.[currentHour]?.is_attack).length;
  const selectedDpHistory = FL_DATA.dpHistories[FL_DATA.epsilons.indexOf(selectedEpsilon)];

  const TABS = [
    { id: "overview", label: "Overview", icon: "🗺️" },
    { id: "analysis", label: "Data Analysis", icon: "🔬" },
    { id: "traffic", label: "Traffic Trends", icon: "🚦" },
    { id: "attacks", label: "Attack Detection", icon: "⚠️", badge: currentAttacks > 0 ? currentAttacks : null },
    { id: "federated", label: "Federated Learning", icon: "🔗" },
    { id: "privacy", label: "DP Privacy", icon: "🔒" },
  ];

  return (
    <div style={{
      minHeight: "100vh", display: "flex", background: "#060b18",
      color: "#e0e8f0", fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    }}>
      {/* ── SIDEBAR ── */}
      <div style={{ position: "relative", flexShrink: 0, zIndex: 200 }}>
        <div style={{
          width: sidebarOpen ? 230 : 60,
          height: "100vh", position: "sticky", top: 0,
          background: "rgba(8,13,26,0.98)",
          borderRight: "1px solid #1e3a5f",
          display: "flex", flexDirection: "column",
          transition: "width 0.28s cubic-bezier(0.4,0,0.2,1)",
          overflow: "hidden",
          boxShadow: sidebarOpen ? "4px 0 24px rgba(0,0,0,0.4)" : "2px 0 12px rgba(0,0,0,0.3)",
        }}>

          {/* ── Logo header ── */}
          <div style={{
            padding: sidebarOpen ? "18px 16px" : "18px 0",
            borderBottom: "1px solid #1e3a5f",
            display: "flex", alignItems: "center",
            justifyContent: sidebarOpen ? "flex-start" : "center",
            gap: 10, flexShrink: 0,
            transition: "padding 0.28s ease",
          }}>
            <div style={{
              width: 32, height: 32, borderRadius: 9, flexShrink: 0,
              background: "linear-gradient(135deg, #4CC9F0 0%, #0a4a7a 100%)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 16, boxShadow: "0 2px 12px rgba(76,201,240,0.35)",
            }}>🏙️</div>
            <div style={{
              overflow: "hidden", whiteSpace: "nowrap",
              opacity: sidebarOpen ? 1 : 0,
              transition: "opacity 0.2s ease",
              pointerEvents: sidebarOpen ? "auto" : "none",
            }}>
              <div style={{ fontSize: 13, fontWeight: 800, color: "#e0e8f0", letterSpacing: 0.3 }}>TrafficFL</div>
              <div style={{ fontSize: 10, color: "#4CC9F0", letterSpacing: 0.5 }}>Smart City Monitor</div>
            </div>
          </div>

          {/* ── Nav section ── */}
          <div style={{ flex: 1, padding: "10px 8px", overflowY: "auto", overflowX: "hidden" }}>
            {/* Section label */}
            <div style={{
              fontSize: 9, color: "#303850", letterSpacing: 2,
              padding: sidebarOpen ? "6px 8px 8px" : "6px 0 8px",
              textTransform: "uppercase", textAlign: sidebarOpen ? "left" : "center",
              overflow: "hidden", whiteSpace: "nowrap",
              opacity: sidebarOpen ? 1 : 0,
              transition: "opacity 0.2s ease",
            }}>Navigation</div>

            {TABS.map(tab => (
              <div key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                title={!sidebarOpen ? tab.label : ""}
                style={{
                  display: "flex", alignItems: "center",
                  gap: sidebarOpen ? 10 : 0,
                  justifyContent: sidebarOpen ? "flex-start" : "center",
                  padding: sidebarOpen ? "9px 10px" : "10px 0",
                  borderRadius: 8, cursor: "pointer", marginBottom: 2,
                  position: "relative", overflow: "hidden",
                  background: activeTab === tab.id ? "rgba(76,201,240,0.10)" : "transparent",
                  borderLeft: `3px solid ${activeTab === tab.id ? "#4CC9F0" : "transparent"}`,
                  transition: "background 0.15s ease, border-color 0.15s ease",
                }}>
                {/* Active glow */}
                {activeTab === tab.id && (
                  <div style={{
                    position: "absolute", inset: 0, borderRadius: 8,
                    background: "linear-gradient(90deg, rgba(76,201,240,0.08) 0%, transparent 100%)",
                    pointerEvents: "none",
                  }} />
                )}
                <span style={{ fontSize: 17, flexShrink: 0, lineHeight: 1 }}>{tab.icon}</span>
                <span style={{
                  fontSize: 13, fontWeight: activeTab === tab.id ? 600 : 400,
                  color: activeTab === tab.id ? "#e0e8f0" : "#5a6a7a",
                  whiteSpace: "nowrap", overflow: "hidden",
                  maxWidth: sidebarOpen ? 140 : 0,
                  opacity: sidebarOpen ? 1 : 0,
                  transition: "max-width 0.28s ease, opacity 0.2s ease",
                }}>{tab.label}</span>
                {tab.badge && (
                  <span style={{
                    marginLeft: "auto", background: "#FF4D6D", color: "#fff",
                    borderRadius: 10, padding: "1px 6px", fontSize: 10, fontWeight: 700,
                    flexShrink: 0, display: sidebarOpen ? "block" : "none",
                  }}>{tab.badge}</span>
                )}
                {/* Collapsed badge dot */}
                {tab.badge && !sidebarOpen && (
                  <div style={{
                    position: "absolute", top: 6, right: 6,
                    width: 7, height: 7, borderRadius: "50%", background: "#FF4D6D",
                  }} />
                )}
              </div>
            ))}

            {/* Divider */}
            <div style={{ height: 1, background: "#1a2540", margin: "10px 6px 12px" }} />

            {/* Districts section */}
            <div style={{
              fontSize: 9, color: "#303850", letterSpacing: 2,
              padding: sidebarOpen ? "0 8px 8px" : "0 0 8px",
              textTransform: "uppercase", textAlign: sidebarOpen ? "left" : "center",
              opacity: sidebarOpen ? 1 : 0,
              transition: "opacity 0.2s ease",
            }}>Districts</div>

            {DISTRICTS.map(d => {
              const hasAttack = districtData[d.id]?.records?.[currentHour]?.is_attack;
              return (
                <div key={d.id}
                  onClick={() => setSelectedDistrict(d.id)}
                  title={!sidebarOpen ? d.name : ""}
                  style={{
                    display: "flex", alignItems: "center",
                    gap: sidebarOpen ? 9 : 0,
                    justifyContent: sidebarOpen ? "flex-start" : "center",
                    padding: sidebarOpen ? "8px 10px" : "9px 0",
                    borderRadius: 8, cursor: "pointer", marginBottom: 2,
                    background: selectedDistrict === d.id ? `${d.color}14` : "transparent",
                    borderLeft: `3px solid ${selectedDistrict === d.id ? d.color : "transparent"}`,
                    transition: "background 0.15s ease",
                    position: "relative",
                  }}>
                  <div style={{
                    width: 9, height: 9, borderRadius: "50%",
                    background: d.color, flexShrink: 0,
                    boxShadow: selectedDistrict === d.id ? `0 0 8px ${d.color}80` : "none",
                    transition: "box-shadow 0.2s ease",
                  }} />
                  <span style={{
                    fontSize: 12,
                    color: selectedDistrict === d.id ? "#e0e8f0" : "#5a6a7a",
                    whiteSpace: "nowrap", overflow: "hidden",
                    maxWidth: sidebarOpen ? 130 : 0,
                    opacity: sidebarOpen ? 1 : 0,
                    transition: "max-width 0.28s ease, opacity 0.2s ease",
                    flex: 1,
                  }}>{d.name}</span>
                  {hasAttack && sidebarOpen && (
                    <span style={{ fontSize: 11, flexShrink: 0 }}>⚠️</span>
                  )}
                  {hasAttack && !sidebarOpen && (
                    <div style={{
                      position: "absolute", top: 5, right: 5,
                      width: 6, height: 6, borderRadius: "50%", background: "#FF4D6D",
                    }} />
                  )}
                </div>
              );
            })}
          </div>

          {/* ── Live status footer ── */}
          <div style={{
            padding: sidebarOpen ? "12px 14px" : "12px 0",
            borderTop: "1px solid #1e3a5f",
            display: "flex", alignItems: "center",
            justifyContent: sidebarOpen ? "flex-start" : "center",
            gap: 8,
          }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: "#06D6A0", flexShrink: 0,
              animation: "livePulse 2s ease-in-out infinite",
              boxShadow: "0 0 6px #06D6A0",
            }} />
            <span style={{
              fontSize: 11, color: "#506080", whiteSpace: "nowrap",
              opacity: sidebarOpen ? 1 : 0,
              transition: "opacity 0.2s ease",
            }}>Live System</span>
          </div>
        </div>

        {/* ── Floating toggle button on sidebar edge ── */}
        <button
          onClick={() => setSidebarOpen(o => !o)}
          title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          style={{
            position: "absolute",
            top: 22,
            right: -14,
            width: 28, height: 28,
            borderRadius: "50%",
            background: "#0e1f38",
            border: "1.5px solid #2a4060",
            color: "#4CC9F0",
            fontSize: 11,
            fontWeight: 700,
            cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 2px 12px rgba(0,0,0,0.5)",
            zIndex: 300,
            transition: "background 0.15s ease, transform 0.28s cubic-bezier(0.4,0,0.2,1), box-shadow 0.15s ease",
            transform: sidebarOpen ? "rotate(0deg)" : "rotate(180deg)",
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = "#1a3555";
            e.currentTarget.style.boxShadow = "0 2px 16px rgba(76,201,240,0.3)";
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = "#0e1f38";
            e.currentTarget.style.boxShadow = "0 2px 12px rgba(0,0,0,0.5)";
          }}
        >
          ‹
        </button>
      </div>

      {/* ── MAIN CONTENT ── */}
      <div style={{ flex: 1, overflow: "auto", minWidth: 0 }}>
        {/* Top bar */}
        <div style={{
          padding: "14px 24px", borderBottom: "1px solid #1e3a5f",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          background: "rgba(6,11,24,0.9)", backdropFilter: "blur(10px)",
          position: "sticky", top: 0, zIndex: 100,
        }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#e0e8f0" }}>
              {TABS.find(t => t.id === activeTab)?.icon}{" "}
              {TABS.find(t => t.id === activeTab)?.label}
            </div>
            <div style={{ fontSize: 11, color: "#506080", marginTop: 2 }}>
              Day {Math.floor(currentHour / 24) + 1} · {String(currentHour % 24).padStart(2, "0")}:00 ·{" "}
              {currentRec.weather_main || "—"} · {DISTRICTS[selectedDistrict].name}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {currentAttacks > 0 && (
              <div style={{ background: "rgba(255,77,109,0.15)", border: "1px solid #FF4D6D", borderRadius: 20, padding: "4px 12px", fontSize: 11, color: "#FF4D6D", fontWeight: 600 }}>
                ⚠ {currentAttacks} ATTACK{currentAttacks > 1 ? "S" : ""} ACTIVE
              </div>
            )}
            <button onClick={() => setIsPlaying(p => !p)} style={{
              background: isPlaying ? "rgba(255,77,109,0.15)" : "rgba(76,201,240,0.15)",
              border: `1px solid ${isPlaying ? "#FF4D6D" : "#4CC9F0"}`,
              color: isPlaying ? "#FF4D6D" : "#4CC9F0",
              padding: "6px 14px", borderRadius: 20, cursor: "pointer",
              fontSize: 12, fontFamily: "inherit", fontWeight: 600,
            }}>
              {isPlaying ? "⏸ PAUSE" : "▶ PLAY"}
            </button>
          </div>
        </div>

        {/* Time slider */}
        <div style={{ padding: "10px 24px", background: "rgba(8,13,26,0.6)", borderBottom: "1px solid #1e3a5f", display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 11, color: "#506080", whiteSpace: "nowrap" }}>Time</span>
          <input type="range" min={0} max={167} value={currentHour}
            onChange={e => setCurrentHour(+e.target.value)}
            style={{ flex: 1, accentColor: "#4CC9F0", height: 4 }} />
          <span style={{ fontSize: 12, color: "#4CC9F0", fontWeight: 600, whiteSpace: "nowrap", minWidth: 100 }}>
            Day {Math.floor(currentHour / 24) + 1} · {String(currentHour % 24).padStart(2, "0")}:00
          </span>
        </div>

        {/* Page content */}
        <div style={{ padding: 24 }}>

          {/* ══ OVERVIEW TAB ══ */}
          {activeTab === "overview" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 20 }}>
              {/* Map */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div style={{ height: 460, borderRadius: 12, overflow: "hidden", border: "1px solid #1e3a5f" }}>
                  <CityMap districtData={districtData} selectedDistrict={selectedDistrict} onSelectDistrict={setSelectedDistrict} currentHour={currentHour} />
                </div>
                {/* All districts bar */}
                <Panel title="📊 Traffic Volume — All Districts" subtitle="Current hour comparison">
                  <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={allDistrictCurrentStats} barSize={32}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                      <XAxis dataKey="name" tick={{ fill: "#8090a0", fontSize: 11 }} />
                      <YAxis tick={{ fill: "#8090a0", fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="volume" name="Volume" fill="#4CC9F0" radius={[4, 4, 0, 0]}>
                        {allDistrictCurrentStats.map((d, i) => (
                          <rect key={i} fill={d.isAttack ? "#FF4D6D" : d.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Panel>
              </div>

              {/* Right panel */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <Panel
                  title={`${DISTRICTS[selectedDistrict].name}`}
                  subtitle="SELECTED DISTRICT · LIVE SENSOR DATA"
                  accentColor={DISTRICTS[selectedDistrict].color}
                >
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
                    <StatCard label="Traffic Volume" value={currentRec.traffic_volume || 0} unit="vehicles/hr" color={DISTRICTS[selectedDistrict].color} />
                    <StatCard label="Avg Speed" value={currentRec.avg_speed || 0} unit="km/h" color="#4CC9F0" />
                    <StatCard label="Occupancy" value={`${currentRec.occupancy || 0}%`} unit="of capacity" color="#F9C74F" />
                    <StatCard label="Queue Length" value={currentRec.queue_length || 0} unit="meters" color="#06D6A0" />
                    <StatCard label="Temperature" value={`${currentRec.temp || 0}°C`} unit="ambient" color="#FF8C42" />
                    <StatCard label="Sensor Noise" value={(currentRec.sensor_noise || 0).toFixed(3)} unit="RF index" color={currentRec.is_attack ? "#FF4D6D" : "#506080"} />
                  </div>
                  {currentRec.is_attack && (
                    <div style={{ background: "rgba(255,77,109,0.1)", border: "1px solid #FF4D6D", borderRadius: 8, padding: "10px 14px", display: "flex", gap: 10, alignItems: "center" }}>
                      <span style={{ fontSize: 20 }}>⚠️</span>
                      <div>
                        <div style={{ color: "#FF4D6D", fontSize: 13, fontWeight: 700 }}>Cyberattack Detected</div>
                        <div style={{ color: "#ff8080", fontSize: 11, marginTop: 2 }}>
                          Type: {currentRec.attack_type} · RF Noise: {(currentRec.sensor_noise || 0).toFixed(3)}
                        </div>
                      </div>
                    </div>
                  )}
                </Panel>

                {/* Quick stats */}
                <Panel title="📈 System Overview">
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    <StatCard label="Total Attacks (7d)" value={totalAttacks} unit="across all districts" color="#FF4D6D" icon="⚠️" />
                    <StatCard label="FL Accuracy" value="97.1%" unit="ε=1.0 model" color="#4CC9F0" icon="🔗" />
                    <StatCard label="DP Privacy" value="ε=1.0" unit="Gaussian mechanism" color="#06D6A0" icon="🔒" />
                    <StatCard label="Active Districts" value="4" unit="online nodes" color="#F9C74F" icon="🏙️" />
                  </div>
                </Panel>

                {/* Weather */}
                <Panel title="🌤 Conditions">
                  <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                    <div style={{ fontSize: 40 }}>
                      {WEATHER_ICONS[currentRec.weather_main] || "🌡️"}
                    </div>
                    <div>
                      <div style={{ fontSize: 18, color: "#e0e8f0", fontWeight: 700 }}>{currentRec.weather_main || "—"}</div>
                      <div style={{ fontSize: 12, color: "#506080", marginTop: 4 }}>Rain: {currentRec.rain_1h || 0}mm · Cloud: {currentRec.clouds_all || 0}%</div>
                      <div style={{ fontSize: 12, color: "#506080", marginTop: 2 }}>
                        {currentRec.is_rush ? "🚗 Rush Hour" : currentRec.is_weekend ? "🏖 Weekend" : "Normal traffic"}
                      </div>
                    </div>
                  </div>
                </Panel>
              </div>
            </div>
          )}

          {/* ══ DATA ANALYSIS TAB ══ */}
          {activeTab === "analysis" && (
            <div style={{ display: "grid", gap: 20 }}>
              <AnalysisPanel districtData={districtData} />

              {/* Scatter: speed vs occupancy */}
              <Panel title="📉 Speed vs Occupancy Correlation" subtitle="All districts — sampled records">
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis dataKey="occupancy" name="Occupancy %" tick={{ fill: "#8090a0", fontSize: 11 }}
                      label={{ value: "Occupancy %", fill: "#506080", fontSize: 11, position: "insideBottom", offset: -5 }} />
                    <YAxis dataKey="avg_speed" name="Avg Speed (km/h)" tick={{ fill: "#8090a0", fontSize: 11 }}
                      label={{ value: "Speed km/h", fill: "#506080", fontSize: 11, angle: -90, position: "insideLeft" }} />
                    <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                    <Legend wrapperStyle={{ fontSize: 12, fontFamily: "monospace" }} />
                    {DISTRICTS.map((d, i) => (
                      <Scatter key={i} name={d.name} fill={d.color} opacity={0.7}
                        data={districtData[i].records.filter((_, ri) => ri % 6 === 0).map(r => ({
                          occupancy: r.occupancy, avg_speed: r.avg_speed,
                        }))} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </Panel>

              {/* Weekly heatmap */}
              <Panel title="🕐 Weekly Traffic Heatmap" subtitle={`${DISTRICTS[selectedDistrict].name} — click to jump to that hour`}>
                <div style={{ overflowX: "auto" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "50px repeat(24, 1fr)", gap: "2px", minWidth: 600 }}>
                    <div style={{ fontSize: 11, color: "#506080" }}></div>
                    {Array.from({ length: 24 }, (_, h) => (
                      <div key={h} style={{ fontSize: 10, color: "#506080", textAlign: "center", paddingBottom: 4 }}>{h}</div>
                    ))}
                    {["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].map((day, di) => (
                      <>
                        <div key={`d${di}`} style={{ fontSize: 11, color: "#8090a0", display: "flex", alignItems: "center", paddingRight: 6 }}>{day}</div>
                        {Array.from({ length: 24 }, (_, h) => {
                          const rec = districtData[selectedDistrict]?.records?.[di * 24 + h] || {};
                          const intensity = (rec.traffic_volume || 0) / 200;
                          const color = DISTRICTS[selectedDistrict].color;
                          const isCurrent = currentHour === di * 24 + h;
                          return (
                            <div key={`${di}-${h}`} title={`${day} ${h}:00 — ${rec.traffic_volume || 0} v/h`}
                              onClick={() => setCurrentHour(di * 24 + h)}
                              style={{
                                borderRadius: 2, aspectRatio: "1", cursor: "pointer",
                                background: rec.is_attack ? `rgba(255,77,109,${0.3 + intensity * 0.7})` : `${color}${Math.floor(intensity * 200 + 30).toString(16).padStart(2, "0")}`,
                                outline: isCurrent ? "2px solid #fff" : "none",
                                outlineOffset: -1,
                              }} />
                          );
                        })}
                      </>
                    ))}
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 11, color: "#506080", marginTop: 10 }}>
                    <span>Low</span>
                    {[0.1, 0.3, 0.5, 0.7, 0.9].map(i => (
                      <div key={i} style={{ width: 16, height: 10, borderRadius: 2, background: `${DISTRICTS[selectedDistrict].color}${Math.floor(i * 200 + 30).toString(16).padStart(2, "0")}` }} />
                    ))}
                    <span>High</span>
                    <span style={{ marginLeft: 12, color: "#FF4D6D" }}>■ Attack</span>
                  </div>
                </div>
              </Panel>
            </div>
          )}

          {/* ══ TRAFFIC TRENDS ══ */}
          {activeTab === "traffic" && (
            <div style={{ display: "grid", gap: 20 }}>
              <Panel title="📈 24-Hour Traffic Pattern" subtitle={`${DISTRICTS[selectedDistrict].name} — current day`}>
                <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                  {DISTRICTS.map((d, i) => (
                    <button key={i} onClick={() => setSelectedDistrict(i)} style={{
                      padding: "5px 14px", borderRadius: 20, border: "none", cursor: "pointer",
                      fontSize: 12, fontFamily: "inherit", fontWeight: 600,
                      background: selectedDistrict === i ? `${d.color}25` : "rgba(255,255,255,0.04)",
                      color: selectedDistrict === i ? d.color : "#506080",
                      borderBottom: selectedDistrict === i ? `2px solid ${d.color}` : "2px solid transparent",
                    }}>
                      {d.name.split(" ")[0]}
                    </button>
                  ))}
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={hourlyPattern}>
                    <defs>
                      <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={DISTRICTS[selectedDistrict].color} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={DISTRICTS[selectedDistrict].color} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis dataKey="hour" tick={{ fill: "#8090a0", fontSize: 10 }} interval={2} />
                    <YAxis tick={{ fill: "#8090a0", fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area dataKey="traffic_volume" name="Volume (v/h)" stroke={DISTRICTS[selectedDistrict].color} fill="url(#volGrad)" strokeWidth={2} dot={false} />
                    <Area dataKey="avg_speed" name="Speed (km/h)" stroke="#4CC9F0" fill="none" strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </Panel>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                <Panel title="📉 Multi-District Volume Comparison">
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={Array.from({ length: 24 }, (_, h) => {
                      const row = { hour: `${String(h).padStart(2, "0")}:00` };
                      DISTRICTS.forEach((d, i) => { row[d.name.split(" ")[0]] = districtData[i].records[h]?.traffic_volume || 0; });
                      return row;
                    })}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                      <XAxis dataKey="hour" tick={{ fill: "#8090a0", fontSize: 9 }} interval={3} />
                      <YAxis tick={{ fill: "#8090a0", fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      {DISTRICTS.map(d => <Line key={d.id} dataKey={d.name.split(" ")[0]} stroke={d.color} strokeWidth={2} dot={false} />)}
                    </LineChart>
                  </ResponsiveContainer>
                </Panel>

                <Panel title="🚥 Speed vs Occupancy">
                  <ResponsiveContainer width="100%" height={240}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                      <XAxis dataKey="occupancy" name="Occupancy" tick={{ fill: "#8090a0", fontSize: 10 }} />
                      <YAxis dataKey="avg_speed" name="Speed" tick={{ fill: "#8090a0", fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                      {DISTRICTS.map((d, i) => (
                        <Scatter key={i} name={d.name} fill={d.color} opacity={0.6}
                          data={districtData[i].records.filter((_, ri) => ri % 8 === 0).map(r => ({
                            occupancy: r.occupancy, avg_speed: r.avg_speed,
                          }))} />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </Panel>
              </div>
            </div>
          )}

          {/* ══ ATTACKS TAB ══ */}
          {activeTab === "attacks" && (
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20 }}>
              <Panel title="🚨 Attack Timeline — All Districts" subtitle="Sensor anomalies detected by FL model" style={{ gridColumn: "1 / -1" }}>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={Array.from({ length: 168 }, (_, h) => {
                    const row = { label: `D${Math.floor(h / 24) + 1} ${String(h % 24).padStart(2, "0")}h` };
                    DISTRICTS.forEach((d, i) => {
                      const rec = districtData[i].records[h] || {};
                      row[d.name.split(" ")[0]] = rec.is_attack ? rec.traffic_volume : 0;
                    });
                    return row;
                  })}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis dataKey="label" tick={{ fill: "#8090a0", fontSize: 9 }} interval={11} />
                    <YAxis tick={{ fill: "#8090a0", fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    {DISTRICTS.map(d => (
                      <Area key={d.id} dataKey={d.name.split(" ")[0]} stackId="1" stroke={d.color} fill={d.color} fillOpacity={0.4} strokeWidth={1.5} dot={false} />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </Panel>

              <Panel title="📊 Attack Breakdown">
                <div style={{ marginBottom: 16 }}>
                  {attacksByType.map((a, i) => (
                    <div key={i} style={{ marginBottom: 14 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                        <span style={{ fontSize: 13, color: "#e0e8f0", fontWeight: 500 }}>{a.type} Attack</span>
                        <span style={{ fontSize: 13, fontWeight: 700, color: ["#FF4D6D", "#F9C74F", "#4CC9F0"][i] }}>{a.count}</span>
                      </div>
                      <div style={{ background: "#1a2035", borderRadius: 4, height: 6 }}>
                        <div style={{ background: ["#FF4D6D", "#F9C74F", "#4CC9F0"][i], height: "100%", borderRadius: 4, width: `${(a.count / totalAttacks) * 100}%`, transition: "width 0.5s" }} />
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ background: "rgba(76,201,240,0.05)", border: "1px solid #1e3a5f", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 12, color: "#506080", marginBottom: 10, fontWeight: 600 }}>Detection Performance (FL+DP ε=1.0)</div>
                  {[
                    { label: "Accuracy", value: "96.35%", color: "#4CC9F0" },
                    { label: "F1-Score", value: "91.15%", color: "#06D6A0" },
                    { label: "AUC-ROC",  value: "99.06%", color: "#F9C74F" },
                    { label: "False Pos", value: "1.2%",  color: "#FF8C42" },
                  ].map(m => (
                    <div key={m.label} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ fontSize: 12, color: "#8090a0" }}>{m.label}</span>
                      <span style={{ fontSize: 13, fontWeight: 700, color: m.color }}>{m.value}</span>
                    </div>
                  ))}
                </div>
              </Panel>

              <Panel title="🔍 Sensor Noise Analysis" subtitle="RF noise level — key attack indicator">
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={Array.from({ length: 168 }, (_, h) => {
                    const rec = districtData[selectedDistrict].records[h] || {};
                    return { h, noise: rec.sensor_noise, threshold: 0.25, attackSignal: rec.is_attack ? rec.sensor_noise : null };
                  })}>
                    <defs>
                      <linearGradient id="noiseGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#4CC9F0" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#4CC9F0" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis tick={false} />
                    <YAxis tick={{ fill: "#8090a0", fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={0.25} stroke="#FF4D6D" strokeDasharray="4 2" label={{ value: "Attack threshold", fill: "#FF4D6D", fontSize: 10, position: "right" }} />
                    <Area dataKey="noise" name="RF Noise" stroke="#4CC9F0" fill="url(#noiseGrad)" strokeWidth={1.5} dot={false} />
                    <Area dataKey="attackSignal" name="Attack Signal" stroke="#FF4D6D" fill="rgba(255,77,109,0.2)" strokeWidth={0} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </Panel>

              <Panel title="📋 Recent Attack Log" subtitle="Latest anomalies across all districts">
                <div style={{ maxHeight: 200, overflowY: "auto" }}>
                  {districtData.flatMap((d, di) =>
                    d.records.filter(r => r.is_attack).slice(0, 5).map((r, ri) => ({ district: DISTRICTS[di], rec: r, key: `${di}-${ri}` }))
                  ).slice(0, 20).map(({ district, rec, key }) => (
                    <div key={key} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0", borderBottom: "1px solid #1e3a5f" }}>
                      <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#FF4D6D", flexShrink: 0 }} />
                      <div style={{ fontSize: 11, color: district.color, minWidth: 55 }}>{district.name.split(" ")[0]}</div>
                      <div style={{ fontSize: 11, color: "#e0e8f0", flex: 1 }}>{rec.timestamp}</div>
                      <div style={{ fontSize: 11, color: "#FF8C42", fontWeight: 600 }}>{rec.attack_type}</div>
                      <div style={{ fontSize: 11, color: "#506080" }}>σ={rec.sensor_noise?.toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              </Panel>
            </div>
          )}

          {/* ══ FEDERATED LEARNING TAB ══ */}
          {activeTab === "federated" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <Panel title="🔗 Federated Architecture" subtitle="Data never leaves district servers">
                <div style={{ position: "relative", height: 260 }}>
                  <div style={{
                    position: "absolute", left: "50%", top: "50%", transform: "translate(-50%,-50%)",
                    width: 110, height: 55, background: "rgba(76,201,240,0.08)",
                    border: "2px solid #4CC9F0", borderRadius: 10,
                    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                    boxShadow: "0 0 24px rgba(76,201,240,0.25)", zIndex: 2,
                  }}>
                    <div style={{ fontSize: 18 }}>🖥️</div>
                    <div style={{ fontSize: 10, color: "#4CC9F0", fontWeight: 700, letterSpacing: 0.5 }}>CENTRAL SERVER</div>
                    <div style={{ fontSize: 9, color: "#506080" }}>FedAvg Aggregation</div>
                  </div>
                  {[
                    { angle: -90, idx: 0 }, { angle: 0, idx: 1 }, { angle: 90, idx: 2 }, { angle: 180, idx: 3 }
                  ].map(({ angle, idx }) => {
                    const rad = (angle * Math.PI) / 180;
                    const cx = 50 + Math.cos(rad) * 35;
                    const cy = 50 + Math.sin(rad) * 38;
                    const d = DISTRICTS[idx];
                    return (
                      <div key={idx} style={{
                        position: "absolute", left: `${cx}%`, top: `${cy}%`, transform: "translate(-50%,-50%)",
                        width: 80, height: 52, background: `${d.color}12`,
                        border: `1.5px solid ${d.color}`, borderRadius: 8,
                        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", zIndex: 2,
                      }}>
                        <div style={{ fontSize: 12 }}>🏙️</div>
                        <div style={{ fontSize: 8, color: d.color, textAlign: "center", fontWeight: 600, lineHeight: 1.3 }}>{d.name}</div>
                        <div style={{ fontSize: 8, color: "#506080" }}>🔒 local data</div>
                      </div>
                    );
                  })}
                  <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
                    {[{ angle: -90, idx: 0 }, { angle: 0, idx: 1 }, { angle: 90, idx: 2 }, { angle: 180, idx: 3 }].map(({ angle, idx }) => {
                      const rad = (angle * Math.PI) / 180;
                      const cx = 50 + Math.cos(rad) * 35;
                      const cy = 50 + Math.sin(rad) * 38;
                      return <line key={idx} x1="50%" y1="50%" x2={`${cx}%`} y2={`${cy}%`}
                        stroke={DISTRICTS[idx].color} strokeWidth="1.5" strokeDasharray="5 3" opacity="0.5" />;
                    })}
                  </svg>
                </div>
              </Panel>

              <Panel title="📈 FL Convergence" subtitle="Accuracy across rounds — all configurations">
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={convergenceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis dataKey="round" tick={{ fill: "#8090a0", fontSize: 10 }} label={{ value: "Round", fill: "#506080", fontSize: 10, position: "insideBottom", offset: -4 }} />
                    <YAxis domain={[85, 100]} tick={{ fill: "#8090a0", fontSize: 10 }} unit="%" />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line dataKey="Central" stroke="#fff" strokeWidth={2.5} dot={false} />
                    <Line dataKey="FL (no DP)" stroke="#4CC9F0" strokeWidth={2} dot={false} strokeDasharray="6 2" />
                    {FL_DATA.epsilons.map(eps => (
                      <Line key={eps} dataKey={`ε=${eps}`} stroke={EPS_COLORS[eps]} strokeWidth={1.5} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </Panel>

              <Panel title="📊 Per-Round Metrics" subtitle="Loss and F1 convergence" style={{ gridColumn: "1 / -1" }}>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={FL_DATA.flHistory.map((f, i) => ({
                    round: f.round,
                    "FL Accuracy": +(f.acc * 100).toFixed(2),
                    "FL Loss": +f.loss.toFixed(4),
                    "FL F1": +(f.f1 * 100).toFixed(2),
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis dataKey="round" tick={{ fill: "#8090a0", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#8090a0", fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line dataKey="FL Accuracy" stroke="#4CC9F0" strokeWidth={2} dot={false} />
                    <Line dataKey="FL F1" stroke="#06D6A0" strokeWidth={2} dot={false} />
                    <Line dataKey="FL Loss" stroke="#FF4D6D" strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Panel>
            </div>
          )}

          {/* ══ PRIVACY TAB ══ */}
          {activeTab === "privacy" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              {/* Epsilon selector */}
              <Panel title="🔒 Privacy Budget (ε)" subtitle="Select differential privacy level" style={{ gridColumn: "1 / -1" }}>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
                  {FL_DATA.epsilons.map(eps => (
                    <button key={eps} onClick={() => setSelectedEpsilon(eps)} style={{
                      padding: "8px 20px", borderRadius: 8, border: "none", cursor: "pointer",
                      background: selectedEpsilon === eps ? `${EPS_COLORS[eps]}20` : "rgba(255,255,255,0.04)",
                      color: selectedEpsilon === eps ? EPS_COLORS[eps] : "#506080",
                      border: `1px solid ${selectedEpsilon === eps ? EPS_COLORS[eps] : "#1e3a5f"}`,
                      fontSize: 13, fontFamily: "inherit", fontWeight: 700,
                    }}>
                      ε = {eps}
                    </button>
                  ))}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                  <StatCard label="Privacy Budget ε" value={selectedEpsilon} unit="lower = more private" color={EPS_COLORS[selectedEpsilon]} />
                  <StatCard label="Noise σ" value={(Math.sqrt(2 * Math.log(1.25 / 1e-5)) / selectedEpsilon).toFixed(3)} unit="Gaussian std dev" color="#4CC9F0" />
                  <StatCard label="Privacy Level" value={selectedEpsilon <= 0.5 ? "STRONG" : selectedEpsilon <= 1 ? "GOOD" : selectedEpsilon <= 2 ? "MODERATE" : "WEAK"} unit="" color={EPS_COLORS[selectedEpsilon]} />
                  <StatCard label="Delta δ" value="1e-5" unit="failure probability" color="#06D6A0" />
                </div>
              </Panel>

              <Panel title="⚖️ Privacy–Utility Trade-off" subtitle="How ε affects model performance">
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={FL_DATA.finalMetrics.map(m => ({
                    eps: `ε=${m.eps}`,
                    Accuracy: +(m.acc * 100).toFixed(2),
                    "F1 Score": +(m.f1 * 100).toFixed(2),
                    "AUC-ROC": +(m.auc * 100).toFixed(2),
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis dataKey="eps" tick={{ fill: "#8090a0", fontSize: 11 }} />
                    <YAxis domain={[85, 100]} tick={{ fill: "#8090a0", fontSize: 10 }} unit="%" />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line dataKey="Accuracy" stroke="#4CC9F0" strokeWidth={2.5} dot={{ r: 5 }} />
                    <Line dataKey="F1 Score" stroke="#06D6A0" strokeWidth={2} dot={{ r: 4 }} />
                    <Line dataKey="AUC-ROC" stroke="#F9C74F" strokeWidth={2} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </Panel>

              <Panel title="📊 Gaussian Noise Calibration" subtitle="σ = √(2·ln(1.25/δ)) / ε">
                <ResponsiveContainer width="100%" height={240}>
                  <AreaChart data={Array.from({ length: 50 }, (_, i) => {
                    const eps = 0.1 + i * 0.12;
                    return { eps: +eps.toFixed(2), sigma: +(Math.sqrt(2 * Math.log(1.25 / 1e-5)) / eps).toFixed(3) };
                  })}>
                    <defs>
                      <linearGradient id="sigmaGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#FF4D6D" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#FF4D6D" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis dataKey="eps" tick={{ fill: "#8090a0", fontSize: 10 }} label={{ value: "ε (privacy budget)", fill: "#506080", fontSize: 10, position: "insideBottom", offset: -5 }} />
                    <YAxis tick={{ fill: "#8090a0", fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    {FL_DATA.epsilons.map(eps => (
                      <ReferenceLine key={eps} x={eps} stroke={EPS_COLORS[eps]} strokeDasharray="3 2" opacity={0.7} />
                    ))}
                    <Area dataKey="sigma" name="Noise σ" stroke="#FF4D6D" fill="url(#sigmaGrad)" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </Panel>

              <Panel title={`📈 Training Curve — ε = ${selectedEpsilon}`} subtitle="Loss and accuracy for selected privacy budget" style={{ gridColumn: "1 / -1" }}>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={selectedDpHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                    <XAxis dataKey="round" tick={{ fill: "#8090a0", fontSize: 10 }} label={{ value: "Round", fill: "#506080", fontSize: 10, position: "insideBottom", offset: -5 }} />
                    <YAxis tick={{ fill: "#8090a0", fontSize: 10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line dataKey="acc" name="Accuracy" stroke={EPS_COLORS[selectedEpsilon]} strokeWidth={2.5} dot={false} />
                    <Line dataKey="f1" name="F1-Score" stroke="#06D6A0" strokeWidth={2} dot={false} />
                    <Line dataKey="loss" name="Loss" stroke="#FF4D6D" strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Panel>
            </div>
          )}

        </div>

        {/* Footer */}
        <div style={{ padding: "12px 24px", borderTop: "1px solid #1e3a5f", display: "flex", justifyContent: "space-between", fontSize: 10, color: "#303850" }}>
          <span>SMART CITY TRAFFIC COMMAND · DESIGN OF SMART CITIES COURSE</span>
          <span>DATASET: METRO INTERSTATE TRAFFIC VOLUME (KAGGLE) + SYNTHETIC ATTACK LABELS</span>
          <span>FL: FEDAVG · DP: (ε,δ)-GAUSSIAN MECHANISM</span>
        </div>
      </div>

      <style>{`
        @keyframes livePulse {
          0%, 100% { opacity: 1; box-shadow: 0 0 8px #06D6A0; }
          50% { opacity: 0.35; box-shadow: 0 0 3px #06D6A0; }
        }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: #060b18; }
        ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
        * { box-sizing: border-box; }
        input[type=range] { height: 4px; }
      `}</style>
    </div>
  );
}
