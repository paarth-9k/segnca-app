import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "../lib/api";

export default function LiveDemo() {
  const [samples, setSamples] = useState([]);
  const [selected, setSelected] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.samples().then((d) => setSamples(d.samples)).catch(() => setError("fetch-failed"));
  }, []);

  async function run(idx = selected) {
    setLoading(true);
    setError(null);
    try {
      const r = await api.predict({ sample_idx: idx, noise_sigma: 0, shift_px: 0, nca_runs: 5 });
      setResult(r);
    } catch (e) {
      setError("predict-failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section id="demo" className="max-w-5xl mx-auto px-6 py-24 border-b border-border">
      <div className="route-label mb-3">POST /api/predict</div>
      <h2 className="font-display text-2xl sm:text-3xl mb-2">Run it on a real scan</h2>
      <p className="text-muted text-[15px] max-w-xl mb-10">
        Pick a held-out slice from the hippocampus MRI validation set. Inference runs live in
        your browser tab, against the actual trained checkpoints on the backend.
      </p>

      {error === "fetch-failed" && (
        <div className="card p-4 text-sm text-nca font-mono">
          Can't reach the API at the configured URL. Start the backend (see setup below) and
          reload — this section needs it to do anything.
        </div>
      )}

      {samples.length > 0 && (
        <>
          <div className="flex flex-wrap gap-2 mb-8">
            {samples.map((s) => (
              <button
                key={s.sample_idx}
                onClick={() => {
                  setSelected(s.sample_idx);
                  run(s.sample_idx);
                }}
                className={`w-14 h-14 rounded-md overflow-hidden border transition ${
                  selected === s.sample_idx
                    ? "border-phosphor ring-1 ring-phosphor"
                    : "border-border hover:border-muted"
                }`}
              >
                <img src={s.thumbnail} alt={`slice ${s.sample_idx}`} className="w-full h-full object-cover" />
              </button>
            ))}
            <button
              onClick={() => run()}
              disabled={loading}
              className="ml-2 px-4 h-14 rounded-md border border-border text-sm font-mono hover:border-phosphor hover:text-phosphor transition disabled:opacity-40"
            >
              {loading ? "running…" : "run inference →"}
            </button>
          </div>

          <AnimatePresence mode="wait">
            {result && (
              <motion.div
                key={selected + JSON.stringify(result.nca.dice)}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.35 }}
              >
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                  <ImgCard label="input" src={result.input_image} />
                  <ImgCard label="ground truth" src={result.ground_truth} accent="border-green-500/40" />
                  <ImgCard label="segnca" src={result.nca_prediction} accent="border-nca/50" />
                  <ImgCard label="tinyunet" src={result.unet_prediction} accent="border-unet/50" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <ResultCard name="SegNCA" colorClass="text-nca" data={result.nca} />
                  <ResultCard name="TinyUNet" colorClass="text-unet" data={result.unet} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </section>
  );
}

function ImgCard({ label, src, accent = "border-border" }) {
  return (
    <div className={`card overflow-hidden ${accent}`}>
      <img src={src} alt={label} className="w-full aspect-square object-cover" style={{ imageRendering: "pixelated" }} />
      <div className="px-2 py-1.5 text-[11px] font-mono text-muted uppercase tracking-wide">{label}</div>
    </div>
  );
}

function ResultCard({ name, colorClass, data }) {
  return (
    <div className="card p-4">
      <div className={`font-mono text-sm ${colorClass} mb-3`}>{name}</div>
      <dl className="space-y-1.5 text-sm font-mono">
        <Row k="dice" v={data.dice.toFixed(3)} />
        <Row k="iou" v={data.iou.toFixed(3)} />
        <Row k="latency" v={`${data.latency_ms.toFixed(1)} ms`} />
        <Row k="params" v={data.params.toLocaleString()} />
      </dl>
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="flex justify-between">
      <dt className="text-muted">{k}</dt>
      <dd className="text-text">{v}</dd>
    </div>
  );
}
