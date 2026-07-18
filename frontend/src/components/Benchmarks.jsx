import { useEffect, useState } from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { api } from "../lib/api";

const GRID_COLOR = "#1f2a37";
const TICK = { fill: "#7d8a9a", fontSize: 11, fontFamily: "IBM Plex Mono" };

export default function Benchmarks() {
  const [metrics, setMetrics] = useState(null);
  const [robustness, setRobustness] = useState(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => {});
    api.robustness().then(setRobustness).catch(() => {});
  }, []);

  if (!metrics || !robustness) {
    return (
      <section className="max-w-5xl mx-auto px-6 py-24 border-b border-border">
        <div className="route-label mb-3">GET /api/metrics · /api/robustness</div>
        <p className="text-muted text-sm font-mono">waiting for backend…</p>
      </section>
    );
  }

  const paramsData = [
    { name: "SegNCA", value: metrics.params.nca, fill: "#e0524f" },
    { name: "TinyUNet", value: metrics.params.unet, fill: "#3f7fd6" },
  ];

  const diceData = [
    { name: "SegNCA", value: metrics.clean.nca_dice_mean, fill: "#e0524f" },
    { name: "TinyUNet", value: metrics.clean.unet_dice, fill: "#3f7fd6" },
  ];

  const noiseData = robustness.noise.sigma.map((s, i) => ({
    sigma: s,
    SegNCA: robustness.noise.nca[i],
    TinyUNet: robustness.noise.unet[i],
  }));

  const shiftData = robustness.shift.shift_px.map((s, i) => ({
    shift: s,
    SegNCA: robustness.shift.nca[i],
    TinyUNet: robustness.shift.unet[i],
  }));

  return (
    <section id="benchmarks" className="max-w-5xl mx-auto px-6 py-24 border-b border-border">
      <div className="route-label mb-3">GET /api/metrics · /api/robustness</div>
      <h2 className="font-display text-2xl sm:text-3xl mb-2">The full study</h2>
      <p className="text-muted text-[15px] max-w-xl mb-10">
        Every chart below is the actual data from {metrics.params.ratio}x-parameter-ratio
        comparison — same loss, same optimizer, checkpoint picked at best validation Dice for
        each model.
      </p>

      <div className="grid md:grid-cols-2 gap-8 mb-12">
        <ChartCard title="parameter count (log-scale in effect)">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={paramsData}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
              <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
              <YAxis tick={TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toLocaleString()} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {paramsData.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="clean validation dice">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={diceData}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
              <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toFixed(3)} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {diceData.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="robustness to intensity noise">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={noiseData}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
              <XAxis dataKey="sigma" tick={TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toFixed(3)} />
              <Line type="monotone" dataKey="SegNCA" stroke="#e0524f" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="TinyUNet" stroke="#3f7fd6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="robustness to spatial shift">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={shiftData}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
              <XAxis dataKey="shift" tick={TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => v.toFixed(3)} />
              <Line type="monotone" dataKey="SegNCA" stroke="#e0524f" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="TinyUNet" stroke="#3f7fd6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <div className="card p-5 text-sm text-muted leading-relaxed">
        <span className="text-text font-mono">note —</span> the shift/noise curves above did{" "}
        <em>not</em> reproduce the paper's translation-robustness claim at this scale: TinyUNet
        held up equally well or better here. Most likely cause: at 48×48px, three pooling levels
        already give the U-Net a near-global receptive field, so the positional-bias failure mode
        the paper targets barely gets a chance to appear. Reported as-is rather than adjusted to
        fit the expected narrative — full reasoning in the written report.
      </div>
    </section>
  );
}

function ChartCard({ title, children }) {
  return (
    <div className="card p-5">
      <div className="text-xs font-mono text-muted uppercase tracking-wide mb-3">{title}</div>
      {children}
    </div>
  );
}

const tooltipStyle = {
  background: "#141b24",
  border: "1px solid #1f2a37",
  borderRadius: 8,
  fontFamily: "IBM Plex Mono",
  fontSize: 12,
};
