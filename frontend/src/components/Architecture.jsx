export default function Architecture() {
  return (
    <section id="architecture" className="max-w-5xl mx-auto px-6 py-24 border-b border-border">
      <div className="route-label mb-3">/models/segnca.py</div>
      <h2 className="font-display text-2xl sm:text-3xl mb-2">One rule, applied everywhere</h2>
      <p className="text-muted text-[15px] max-w-xl mb-10">
        SegNCA has no encoder, no decoder, no downsampling. It's a single tiny update rule that
        every cell in the image runs, in parallel, for 28 steps.
      </p>

      <div className="grid md:grid-cols-3 gap-6">
        <Step
          n="input"
          title="perceive"
          body="Each cell reads its own state plus a Sobel-X / Sobel-Y gradient of its 3×3 neighborhood — fixed, non-trainable filters. No cell ever sees anything beyond one ring of neighbors."
        />
        <Step
          n="rule"
          title="update"
          body="Linear(36→128) → ReLU → Linear(128→16). Output-layer weights start at zero, so the untrained rule is a no-op. 8,320 numbers, total."
        />
        <Step
          n="×28"
          title="persist"
          body="The MRI intensity channel is re-injected after every step — it never evolves. Only the hidden + segmentation-logit channels accumulate a decision, step by step, exactly like the underlying tissue is being 're-examined' 28 times."
        />
      </div>

      <div className="mt-10 card p-5 font-mono text-xs text-muted leading-relaxed overflow-x-auto">
        <pre className="whitespace-pre">{`class SegNCA(nn.Module):
    def forward(self, x, steps=28, fire_rate=0.5):
        state = cat([x, zeros(hidden_channels)])
        for _ in range(steps):
            y = perceive(state)                    # sobel x/y, depthwise
            dx = fc1(relu(fc0(y)))                  # the entire rule
            fire = (rand(...) > fire_rate).float()  # async stochastic update
            new_state = state + dx * fire
            state = cat([state[:input_ch], new_state[input_ch:]])  # image pinned
        return state[..., -1]  # segmentation logit`}</pre>
      </div>
    </section>
  );
}

function Step({ n, title, body }) {
  return (
    <div className="card p-5">
      <div className="font-mono text-xs text-phosphor mb-3">{n}</div>
      <div className="font-display text-base mb-2">{title}</div>
      <p className="text-sm text-muted leading-relaxed">{body}</p>
    </div>
  );
}
