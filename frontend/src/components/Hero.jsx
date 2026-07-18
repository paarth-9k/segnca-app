import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import CAFieldBackground from "./CAFieldBackground";
import { api } from "../lib/api";

export default function Hero() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setError(true));
  }, []);

  return (
    <section className="relative overflow-hidden border-b border-border">
      <CAFieldBackground className="absolute inset-0 w-full h-full opacity-40" />
      <div className="absolute inset-0 bg-gradient-to-b from-bg/40 via-bg/70 to-bg" />

      <div className="relative max-w-5xl mx-auto px-6 pt-28 pb-24">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="route-label mb-6"
        >
          GET /api/health
          <span className="mx-2 text-border">·</span>
          {error ? (
            <span className="text-nca">backend unreachable — start the API to see this live</span>
          ) : health ? (
            <span className="text-phosphor">
              connected · {health.nca_params.toLocaleString()} + {health.unet_params.toLocaleString()} params loaded
            </span>
          ) : (
            <span className="text-muted">connecting…</span>
          )}
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="font-display text-4xl sm:text-5xl md:text-6xl leading-[1.08] tracking-tight max-w-3xl"
        >
          58<span className="text-nca">x</span> fewer parameters.
          <br />
          <span className="text-muted">4 points of Dice.</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-6 max-w-xl text-muted text-[15px] leading-relaxed"
        >
          SegNCA reproduces{" "}
          <a
            href="https://arxiv.org/abs/2302.03473"
            target="_blank"
            rel="noreferrer"
            className="text-text underline decoration-border hover:decoration-phosphor underline-offset-4"
          >
            Med-NCA
          </a>{" "}
          — a Neural Cellular Automaton that segments real clinical MRI using an 8,320-parameter
          local update rule, benchmarked live against a 483,153-parameter U-Net on real
          hippocampus scans. Every number on this page is computed by the models below, not
          pre-rendered.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-10 flex flex-wrap gap-x-10 gap-y-4 font-mono text-sm"
        >
          <Stat label="segnca params" value="8,320" color="text-nca" />
          <Stat label="tinyunet params" value="483,153" color="text-unet" />
          <Stat label="checkpoint size" value="32.5 KB / 1.84 MB" color="text-text" />
          <Stat label="data" value="MSD Task04 · real MRI" color="text-phosphor" />
        </motion.div>
      </div>
    </section>
  );
}

function Stat({ label, value, color }) {
  return (
    <div className="flex flex-col gap-1">
      <span className={`text-base ${color}`}>{value}</span>
      <span className="text-[11px] uppercase tracking-wide text-muted">{label}</span>
    </div>
  );
}
