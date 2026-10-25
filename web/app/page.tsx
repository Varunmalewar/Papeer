"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth/client";

function Sparkle({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className={className} aria-hidden>
      <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z" strokeLinejoin="round" />
    </svg>
  );
}

function ArrowUp({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={className} aria-hidden>
      <path d="M12 19V5M6 11l6-6 6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Plus({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={className} aria-hidden>
      <path d="M12 5v14M5 12h14" strokeLinecap="round" />
    </svg>
  );
}

function LogoMark({ className = "h-9 w-9" }: { className?: string }) {
  return (
    <span
      className={`grid place-items-center rounded-xl border border-astra-300/20 bg-gradient-to-br from-astra-400/25 to-astra-600/10 text-astra-200 shadow-[0_0_30px_rgba(91,125,255,0.25)] ${className}`}
    >
      <Sparkle className="h-[55%] w-[55%]" />
    </span>
  );
}

export default function Home() {
  const [email, setEmail] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const router = useRouter();

  useEffect(() => {
    let cancelled = false;
    authClient
      .getSession()
      .then(({ data }) => {
        if (!cancelled) {
          setEmail(data?.user?.email ?? null);
          setLoaded(true);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setEmail(null);
          setLoaded(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (loaded && email) {
      router.push("/chat");
    }
  }, [loaded, email, router]);

  const startHref = email ? "/chat" : "/auth/sign-up";

  return (
    <main className="relative flex min-h-dvh flex-col">
      <header className="flex items-center justify-between px-6 py-5 md:px-10">
        <Link href="/" className="flex items-center gap-2.5">
          <LogoMark className="h-8 w-8" />
          <span className="text-sm font-medium tracking-wide text-slate-100">Papeer</span>
        </Link>
        <nav className="flex items-center gap-2 text-sm">
          {email ? (
            <Link
              href="/chat"
              className="rounded-full bg-white px-4 py-2 text-[13px] font-medium text-[#05070d] transition hover:bg-astra-100"
            >
              Open chat
            </Link>
          ) : (
            <>
              <Link
                href="/auth/sign-in"
                className="rounded-full px-4 py-2 text-[13px] text-slate-300 transition hover:bg-white/5 hover:text-white"
              >
                Sign in
              </Link>
              <Link
                href="/auth/sign-up"
                className="rounded-full bg-white px-4 py-2 text-[13px] font-medium text-[#05070d] transition hover:bg-astra-100"
              >
                Get started
              </Link>
            </>
          )}
        </nav>
      </header>

      <section className="mx-auto flex w-full max-w-6xl flex-1 flex-col justify-center px-6 pb-20 pt-10 md:px-10">
        <p className="text-[11px] font-medium uppercase tracking-[0.32em] text-astra-300/80">
          Research, reimagined
        </p>

        <h1 className="mt-5 max-w-4xl text-5xl font-light leading-[1.04] tracking-tight text-white md:text-7xl">
          Shape your{" "}
          <span className="text-gradient">understanding.</span>
        </h1>

        <p className="mt-6 max-w-xl text-[15px] leading-relaxed text-slate-400 md:text-base">
          Upload papers, ask across them, verify claims and search the web — one
          calm space for research. Every answer cites where it came from.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link
            href={startHref}
            className="rounded-full bg-white px-5 py-2.5 text-sm font-medium text-[#05070d] transition hover:bg-astra-100"
          >
            {loaded && email ? "Open chat" : "Start researching"}
          </Link>
          <Link
            href="/auth/sign-in"
            className="glass rounded-full px-5 py-2.5 text-sm text-slate-200 transition hover:bg-white/10"
          >
            Sign in
          </Link>
        </div>

        <Link
          href={startHref}
          className="glass group mt-12 flex w-full max-w-2xl items-end justify-between gap-5 rounded-[1.6rem] p-4 transition hover:border-astra-300/25 hover:bg-white/[0.07] md:p-5"
        >
          <div className="min-w-0">
            <p className="text-[15px] leading-relaxed text-slate-200">
              Ask a question across your papers — or verify a specific claim.
            </p>
            <div className="mt-3 flex items-center gap-2 text-[11px] text-slate-500">
              <span className="grid h-6 w-6 place-items-center rounded-full border border-white/10 bg-white/5 text-slate-300">
                <Plus className="h-3 w-3" />
              </span>
              <span className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-slate-300">
                <Sparkle className="h-3 w-3 text-astra-300" />
                Papeer RAG
              </span>
            </div>
          </div>
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white text-[#05070d] transition group-hover:scale-105">
            <ArrowUp className="h-4 w-4" />
          </span>
        </Link>

        <div className="mt-14 grid gap-3 text-[13px] text-slate-500 sm:grid-cols-3">
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-astra-400" />
            Grounded in your documents
          </div>
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Live web search when needed
          </div>
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-violet-400" />
            Claim verification with sources
          </div>
        </div>
      </section>

      <footer className="px-6 pb-8 text-[11px] text-slate-600 md:px-10">
        © Papeer · Research paper assistant
      </footer>
    </main>
  );
}
