import Hero from "./components/Hero";
import LiveDemo from "./components/LiveDemo";
import RobustnessPlayground from "./components/RobustnessPlayground";
import Benchmarks from "./components/Benchmarks";
import Architecture from "./components/Architecture";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div className="min-h-screen bg-bg text-text">
      <nav className="sticky top-0 z-20 backdrop-blur bg-bg/80 border-b border-border">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <span className="font-display text-sm">segnca</span>
          <div className="flex gap-6 font-mono text-xs text-muted">
            <a href="#demo" className="hover:text-text">demo</a>
            <a href="#robustness" className="hover:text-text">robustness</a>
            <a href="#benchmarks" className="hover:text-text">benchmarks</a>
            <a href="#architecture" className="hover:text-text">architecture</a>
          </div>
        </div>
      </nav>
      <Hero />
      <LiveDemo />
      <RobustnessPlayground />
      <Benchmarks />
      <Architecture />
      <Footer />
    </div>
  );
}
