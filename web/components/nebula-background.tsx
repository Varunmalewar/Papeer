"use client";

import { useEffect, useRef } from "react";

type Star = {
  x: number;
  y: number;
  z: number;
  r: number;
  phase: number;
  twinkle: number;
  drift: number;
};

/**
 * Fixed ambient background: two soft nebula glows plus a lightweight canvas
 * starfield. Runs at a capped device pixel ratio and pauses when the tab is
 * hidden. Users with reduced-motion get a static frame.
 */
export function NebulaBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let stars: Star[] = [];
    let raf = 0;
    let last = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.min(240, Math.round((width * height) / 8500));
      stars = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        z: 0.35 + Math.random() * 0.65,
        r: 0.35 + Math.random() * 1.45,
        phase: Math.random() * Math.PI * 2,
        twinkle: 0.35 + Math.random() * 1.1,
        drift: (Math.random() - 0.5) * 0.05,
      }));
    };

    const draw = (time: number) => {
      const dt = Math.min(48, time - (last || time));
      last = time;
      ctx.clearRect(0, 0, width, height);

      for (const star of stars) {
        if (!reduced) {
          star.y -= dt * 0.008 * star.z;
          star.x += star.drift * dt * 0.02;
          if (star.y < -12) {
            star.y = height + 12;
            star.x = Math.random() * width;
          }
          if (star.x < -12) star.x = width + 12;
          if (star.x > width + 12) star.x = -12;
        }

        const alpha = 0.28 + 0.6 * (0.5 + 0.5 * Math.sin(time * 0.0011 * star.twinkle + star.phase));
        const radius = star.r * star.z;

        ctx.beginPath();
        ctx.fillStyle = `rgba(190, 214, 255, ${alpha})`;
        ctx.arc(star.x, star.y, radius, 0, Math.PI * 2);
        ctx.fill();

        if (star.r > 1.05) {
          ctx.beginPath();
          ctx.fillStyle = `rgba(160, 195, 255, ${alpha * 0.12})`;
          ctx.arc(star.x, star.y, radius * 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (!reduced) raf = requestAnimationFrame(draw);
    };

    resize();
    raf = requestAnimationFrame(draw);

    const onResize = () => {
      resize();
      if (reduced) draw(0);
    };
    const onVisibility = () => {
      if (document.hidden) {
        cancelAnimationFrame(raf);
      } else if (!reduced) {
        last = 0;
        raf = requestAnimationFrame(draw);
      }
    };

    window.addEventListener("resize", onResize);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[#05070d]">
      <div
        className="absolute -right-[12%] top-[-8%] h-[78vw] w-[78vw] max-h-[1150px] max-w-[1150px] rounded-full"
        style={{
          background:
            "radial-gradient(circle at center, rgba(70, 112, 255, 0.55) 0%, rgba(34, 66, 180, 0.26) 40%, transparent 70%)",
          animation: "nebula-drift 26s ease-in-out infinite alternate",
        }}
      />
      <div
        className="absolute -bottom-[26%] -left-[14%] h-[68vw] w-[68vw] max-h-[1000px] max-w-[1000px] rounded-full"
        style={{
          background:
            "radial-gradient(circle at center, rgba(120, 86, 255, 0.4) 0%, rgba(62, 42, 160, 0.18) 44%, transparent 72%)",
          animation: "nebula-drift 34s ease-in-out infinite alternate-reverse",
        }}
      />
      <div
        className="absolute left-[52%] top-[30%] h-[42vw] w-[42vw] max-h-[700px] max-w-[700px] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          background:
            "radial-gradient(circle at center, rgba(48, 96, 230, 0.3) 0%, transparent 68%)",
        }}
      />
      <div
        className="absolute bottom-[6%] right-[8%] h-[30vw] w-[30vw] max-h-[520px] max-w-[520px] rounded-full opacity-70"
        style={{
          background:
            "radial-gradient(circle at center, rgba(140, 170, 255, 0.16) 0%, transparent 70%)",
        }}
      />
      <canvas ref={canvasRef} className="absolute inset-0" />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[#05070d]/80" />
    </div>
  );
}
