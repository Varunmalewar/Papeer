import Link from "next/link";
import type { ReactNode } from "react";

function Sparkle({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className={className} aria-hidden>
      <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z" strokeLinejoin="round" />
    </svg>
  );
}

export function AuthShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <main className="relative flex min-h-dvh flex-col items-center justify-center px-5 py-12">
      <div className="w-full max-w-md">
        <Link href="/" className="mb-6 flex items-center justify-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-xl border border-astra-300/20 bg-gradient-to-br from-astra-400/25 to-astra-600/10 text-astra-200 shadow-[0_0_30px_rgba(91,125,255,0.25)]">
            <Sparkle className="h-4 w-4" />
          </span>
          <span className="text-sm font-medium tracking-wide text-slate-100">Papeer</span>
        </Link>

        <div className="glass rounded-3xl p-6 shadow-2xl shadow-black/40 md:p-8">
          <h1 className="text-xl font-light tracking-tight text-white">{title}</h1>
          <p className="mt-1.5 text-sm text-slate-400">{subtitle}</p>
          <div className="mt-6">{children}</div>
        </div>

        <p className="mt-5 text-center text-[11px] leading-relaxed text-slate-600">
          Your papers stay private to your account.
        </p>
      </div>
    </main>
  );
}
