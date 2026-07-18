import { useEffect, useState, useCallback, useRef } from "react";
import { api } from "../lib/api";

export default function RobustnessPlayground() {
  const [noise, setNoise] = useState(0);
  const [shift, setShift] = useState(0);
  const [sampleIdx, setSampleIdx] = useState(3);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef(null);

  const run = useCallback((n, s, idx) => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const r = await api.predict({ sample_idx: idx, noise_sigma: n, shift_px: s, nca_runs: 4 });
        setResult(r);
      } catch (e) {
        /* silent - health badge in hero already communicates backend state */
      } finally {
        setLoading(false);
      }
    }, 180);
  }, []);

  useEffect(() => {
    run(noise, shift, sampleIdx);
  }, [noise, shift, sampleIdx, run]);

  return (
    <section id="robustness" className="max-w-5xl mx-auto px-6 py-24 border-b border-border">
      <div className="route-label mb-3">POST /api/predict · sigma, shift_px</div>
      <h2 className="font-display text-2xl sm:text-3xl mb-2">Break it on purpose</h2>
      <p className="text-muted text-[15px] max-w-xl mb-10">
        Drag either slider and the backend re-runs both models on the corrupted slice, live. This
        is the same stress test from the written report — noise simulates acquisition artifacts,
        shift tests whether a purely local update rule really is more position-invariant than a
        CNN with a learned receptive field.
      </p>

      <div className="grid md:grid-cols-2 gap-10 mb-10">
        <Slider
          label="gaussian noise σ"
          value={noise}
          min={0}
          max={0.3}
          step={0.01}
          onChange={setNoise}
          display={noise.toFixed(2)}
        />
        <Slider
          label="spatial shift (px)"
          value={shift}
          min={0}
          max={10}
          step={1}
          onChange={setShift}
          display={`${shift}px`}
        />
      </div>

      {result && (
        <div className={`grid grid-cols-3 gap-4 transition-opacity ${loading ? "opacity-50" : "opacity-100"}`}>
          <div className="card overflow-hidden">
            <img src={result.input_image} className="w-full aspect-square object-cover" style={{ imageRendering: "pixelated" }} />
            <div className="px-2 py-1.5 text-[11px] font-mono text-muted uppercase tracking-wide">corrupted input</div>
          </div>
          <RobustCard label="segnca" accent="border-nca/50" src={result.nca_prediction} dice={result.nca.dice} colorClass="text-nca" />
          <RobustCard label="tinyunet" accent="border-unet/50" src={result.unet_prediction} dice={result.unet.dice} colorClass="text-unet" />
        </div>
      )}
    </section>
  );
}

function Slider({ label, value, min, max, step, onChange, display }) {
  return (
    <div>
      <div className="flex justify-between mb-3 font-mono text-sm">
        <span className="text-muted">{label}</span>
        <span className="text-text">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

function RobustCard({ label, accent, src, dice, colorClass }) {
  return (
    <div className={`card overflow-hidden ${accent}`}>
      <img src={src} className="w-full aspect-square object-cover" style={{ imageRendering: "pixelated" }} />
      <div className="px-2 py-1.5 flex justify-between items-center">
        <span className="text-[11px] font-mono text-muted uppercase tracking-wide">{label}</span>
        <span className={`text-xs font-mono ${colorClass}`}>{dice.toFixed(3)}</span>
      </div>
    </div>
  );
}
