export default function Footer() {
  return (
    <footer className="max-w-5xl mx-auto px-6 py-16">
      <div className="flex flex-col sm:flex-row justify-between gap-6 text-sm">
        <div>
          <div className="font-display text-text mb-2">SegNCA</div>
          <p className="text-muted max-w-sm leading-relaxed">
            A reproduction of Med-NCA (Kalkhof, González &amp; Mukhopadhyay, MICCAI/IPMI 2023),
            trained on real Medical Segmentation Decathlon hippocampus MRI, benchmarked live
            against a standard U-Net.
          </p>
        </div>
        <div className="flex flex-col gap-2 font-mono text-muted">
          <a href="https://arxiv.org/abs/2302.03473" target="_blank" rel="noreferrer" className="hover:text-text">
            → original paper
          </a>
          <a href="https://github.com/MECLabTUDA/M3D-NCA" target="_blank" rel="noreferrer" className="hover:text-text">
            → official implementation
          </a>
          <a href="#demo" className="hover:text-text">→ live demo</a>
          <a href="#benchmarks" className="hover:text-text">→ full study</a>
        </div>
      </div>
      <div className="mt-12 pt-6 border-t border-border text-xs text-muted font-mono">
        data source: Medical Segmentation Decathlon, Task04_Hippocampus (Simpson et al., 2019) · single-volume reproduction, not a clinical benchmark
      </div>
    </footer>
  );
}
