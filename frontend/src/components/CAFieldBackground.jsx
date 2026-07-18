import { useEffect, useRef } from "react";
import weights from "../assets/nca_weights.json";

// Same forward pass as the standalone growing-NCA demo, verified against the
// PyTorch reference to float32 precision. Runs continuously, very low opacity,
// as ambient texture -- this is the actual trained rule, not a decorative loop.
const GRID = 32;
const C = weights.channels;
const H = weights.hidden;
const FIRE_RATE = 0.5;

function idx(y, x, c) {
  return (y * GRID + x) * C + c;
}

function makeEngine() {
  let state = new Float32Array(GRID * GRID * C);
  const reset = () => {
    state.fill(0);
    const cy = GRID >> 1, cx = GRID >> 1;
    for (let c = 3; c < C; c++) state[idx(cy, cx, c)] = 1.0;
  };
  reset();

  function sobelXAt(y, x, c) {
    const k = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= GRID || xx < 0 || xx >= GRID) continue;
      s += k[dy + 1][dx + 1] * state[idx(yy, xx, c)];
    }
    return s / 8.0;
  }
  function sobelYAt(y, x, c) {
    const k = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= GRID || xx < 0 || xx >= GRID) continue;
      s += k[dy + 1][dx + 1] * state[idx(yy, xx, c)];
    }
    return s / 8.0;
  }
  function maxAlphaN(alpha, y, x) {
    let m = -Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= GRID || xx < 0 || xx >= GRID) continue;
      const v = alpha[yy * GRID + xx];
      if (v > m) m = v;
    }
    return m;
  }

  let stepCount = 0;
  function step() {
    const alphaBefore = new Float32Array(GRID * GRID);
    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) alphaBefore[y * GRID + x] = state[idx(y, x, 3)];
    const preAlive = new Uint8Array(GRID * GRID);
    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) preAlive[y * GRID + x] = maxAlphaN(alphaBefore, y, x) > 0.1 ? 1 : 0;

    const perception = new Float32Array(GRID * GRID * 3 * C);
    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) {
      for (let c = 0; c < C; c++) {
        const base = (y * GRID + x) * 3 * C + 3 * c;
        perception[base + 0] = state[idx(y, x, c)];
        perception[base + 1] = sobelXAt(y, x, c);
        perception[base + 2] = sobelYAt(y, x, c);
      }
    }

    const newState = new Float32Array(state.length);
    const fireMask = new Uint8Array(GRID * GRID);
    for (let i = 0; i < GRID * GRID; i++) fireMask[i] = Math.random() <= FIRE_RATE ? 1 : 0;
    const hiddenBuf = new Float32Array(H);
    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) {
      const pbase = (y * GRID + x) * 3 * C;
      for (let h = 0; h < H; h++) {
        let s = weights.b1[h];
        const row = weights.W1[h];
        for (let k = 0; k < 3 * C; k++) s += row[k] * perception[pbase + k];
        hiddenBuf[h] = s > 0 ? s : 0;
      }
      const fire = fireMask[y * GRID + x];
      for (let c = 0; c < C; c++) {
        let s = weights.b2[c];
        const row = weights.W2[c];
        for (let h = 0; h < H; h++) s += row[h] * hiddenBuf[h];
        const cur = state[idx(y, x, c)];
        newState[idx(y, x, c)] = cur + (fire ? s : 0);
      }
    }

    const alphaAfter = new Float32Array(GRID * GRID);
    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) alphaAfter[y * GRID + x] = newState[idx(y, x, 3)];
    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) {
      const postAlive = maxAlphaN(alphaAfter, y, x) > 0.1 ? 1 : 0;
      const alive = preAlive[y * GRID + x] && postAlive ? 1 : 0;
      for (let c = 0; c < C; c++) newState[idx(y, x, c)] = alive ? newState[idx(y, x, c)] : 0;
    }
    state = newState;
    stepCount++;
    if (stepCount > 140) { reset(); stepCount = 0; } // loop: grow, hold, reseed
  }

  return { step, getState: () => state };
}

export default function CAFieldBackground({ className = "" }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const engine = makeEngine();
    let raf;
    let frame = 0;

    function render() {
      const state = engine.getState();
      const img = ctx.createImageData(GRID, GRID);
      for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) {
        const a = Math.min(Math.max(state[idx(y, x, 3)], 0), 1);
        const p = (y * GRID + x) * 4;
        // phosphor-teal tint instead of the demo's red, to read as ambient texture
        img.data[p] = 30 + a * 60;
        img.data[p + 1] = 40 + a * 210;
        img.data[p + 2] = 45 + a * 190;
        img.data[p + 3] = a * 90;
      }
      ctx.putImageData(img, 0, 0);
    }

    function loop() {
      frame++;
      if (frame % 3 === 0) engine.step();
      render();
      raf = requestAnimationFrame(loop);
    }
    loop();
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={GRID}
      height={GRID}
      className={className}
      style={{ imageRendering: "pixelated" }}
      aria-hidden="true"
    />
  );
}
