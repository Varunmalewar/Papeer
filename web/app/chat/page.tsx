"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { authClient } from "@/lib/auth/client";
import {
  apiArxiv,
  apiBtw,
  apiChatStream,
  apiDeleteSessionVectors,
  apiHistory,
  apiPaperChunks,
  apiPapersList,
  apiRename,
  apiUpload,
  apiWebpage,
  type Source,
} from "@/lib/api";

type Msg = {
  role: "user" | "assistant";
  content: string;
  route?: string | null;
  sources?: Source[];
  sideNote?: boolean;
};

/** Models sometimes emit \[...\] / \(...\) LaTeX delimiters; KaTeX needs
 * $$...$$ / $...$. Normalize before parsing. */
function normalizeMath(text: string): string {
  let out = text.replace(/\\\[([\s\S]*?)\\\]/g, (_m, eq) => `\n$$${eq}\n$$\n`);
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_m, eq) => `$${eq}$`);
  return out;
}

function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {normalizeMath(text)}
      </ReactMarkdown>
    </div>
  );
}

type SessionMeta = { id: string; name: string; createdAt: string; isNamed: boolean };
type Flash = { kind: "success" | "error" | "info"; text: string };

// ── icons ────────────────────────────────────────────────────────────────────

function Sparkle({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className={className} aria-hidden>
      <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z" strokeLinejoin="round" />
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

function ArrowUp({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className} aria-hidden>
      <path d="M12 19V5M6 11l6-6 6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Trash({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className={className} aria-hidden>
      <path d="M4 7h16M9 7V5a1 1 0 011-1h4a1 1 0 011 1v2M6 7l1 13h10l1-13" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Menu({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className={className} aria-hidden>
      <path d="M4 7h16M4 12h16M4 17h16" strokeLinecap="round" />
    </svg>
  );
}

function Close({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" className={className} aria-hidden>
      <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
    </svg>
  );
}

function Logout({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className={className} aria-hidden>
      <path d="M15 17l5-5-5-5M20 12H9M13 4H6a2 2 0 00-2 2v12a2 2 0 002 2h7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Doc({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className={className} aria-hidden>
      <path d="M7 3h7l4 4v14H7V3z" strokeLinejoin="round" />
      <path d="M14 3v4h4M10 12h5M10 16h5" strokeLinecap="round" />
    </svg>
  );
}

// ── route badge ──────────────────────────────────────────────────────────────

const ROUTE_BADGE: Record<string, { label: string; color: string }> = {
  paper_qa: { label: "Vector Retrieval", color: "#82a4ff" },
  live_search: { label: "Tavily Web Search", color: "#4ade80" },
  verify_claim: { label: "Claim Verified", color: "#c084fc" },
  direct_answer: { label: "Direct Answer", color: "#fbbf24" },
};

function RouteBadge({ route }: { route?: string | null }) {
  if (!route) return null;
  const cfg = ROUTE_BADGE[route] ?? { label: route, color: "#93a3bf" };
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-medium"
      style={{
        color: cfg.color,
        borderColor: `${cfg.color}40`,
        background: `${cfg.color}14`,
      }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: cfg.color }} />
      {cfg.label}
    </span>
  );
}

// ── localStorage helpers (sessions persist like ChatGPT) ─────────────────────

function lsRead<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function lsWrite(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage full/blocked — ignore
  }
}

function lsRemove(key: string) {
  try {
    localStorage.removeItem(key);
  } catch {
    // ignore
  }
}

const sessionsKey = (email: string) => `papeer.sessions.${email}`;
const activeKey = (email: string) => `papeer.active.${email}`;
const msgsKey = (sid: string) => `papeer.msgs.${sid}`;

/** Stable per-user document library id (survives reloads, logins, new chats). */
async function userLibraryId(mail: string): Promise<string> {
  const normalized = mail.trim().toLowerCase();
  try {
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(normalized));
    const hex = Array.from(new Uint8Array(digest).slice(0, 16), (b) =>
      b.toString(16).padStart(2, "0"),
    ).join("");
    return `lib-${hex}`;
  } catch {
    let h = 2166136261;
    for (let i = 0; i < normalized.length; i++) {
      h ^= normalized.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return `lib-${(h >>> 0).toString(16)}`;
  }
}

export default function ChatPage() {
  const [email, setEmail] = useState<string | null>(null);
  const [libraryId, setLibraryId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [papers, setPapers] = useState<{ titles: string[]; chunks: number } | null>(null);
  const [activeDoc, setActiveDoc] = useState<string | null>(null);
  const [flash, setFlash] = useState<Flash[]>([]);
  const [urlInput, setUrlInput] = useState("");
  const [arxivInput, setArxivInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const quickFileRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  function addFlash(kind: Flash["kind"], text: string) {
    setFlash((f) => [...f.slice(-4), { kind, text }]);
  }

  // ── bootstrap: auth → per-user session list ────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    authClient
      .getSession()
      .then(async ({ data }) => {
        if (cancelled) return;
        const mail = data?.user?.email ?? null;
        setEmail(mail);
        if (mail) {
          const lib = await userLibraryId(mail);
          if (cancelled) return;
          setLibraryId(lib);
          const saved = lsRead<SessionMeta[]>(sessionsKey(mail), []);
          if (saved.length > 0) {
            const lastActive = lsRead<string | null>(activeKey(mail), null);
            const restoreId =
              lastActive && saved.some((s) => s.id === lastActive)
                ? lastActive
                : [...saved].sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0].id;
            setSessions(saved);
            setActiveId(restoreId);
          } else {
            const sid = crypto.randomUUID();
            const meta: SessionMeta = {
              id: sid,
              name: "New Session",
              createdAt: new Date().toISOString(),
              isNamed: false,
            };
            setSessions([meta]);
            lsWrite(sessionsKey(mail), [meta]);
            setActiveId(sid);
          }
        }
      })
      .catch(() => {
        if (!cancelled) setEmail(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // ── persist the last opened session per user ───────────────────────────────
  useEffect(() => {
    if (email && activeId) lsWrite(activeKey(email), activeId);
  }, [email, activeId]);

  /** Vector-store scope: the user's library plus all their chat ids (legacy docs). */
  function libraryScope(): string[] {
    const ids = [libraryId, ...sessions.map((s) => s.id)].filter(Boolean) as string[];
    return Array.from(new Set(ids));
  }

  // ── session open: history restore + documents (user library) ───────────────
  useEffect(() => {
    if (!activeId || !libraryId) return;
    let cancelled = false;
    const scope = Array.from(new Set([libraryId, ...sessions.map((s) => s.id)]));

    setPapers(null);
    const local = lsRead<Msg[] | null>(msgsKey(activeId), null);
    setMsgs(local ?? []);
    if (!local) {
      apiHistory(activeId)
        .then(({ messages }) => {
          if (cancelled) return;
          const restored: Msg[] = messages.map((m) => ({ role: m.role, content: m.content }));
          setMsgs(restored);
          lsWrite(msgsKey(activeId), restored);
        })
        .catch(() => {
          /* no history yet — fresh session */
        });
    }

    Promise.all([apiPapersList(libraryId, scope), apiPaperChunks(libraryId, scope)])
      .then(([list, status]) => {
        if (!cancelled) {
          setPapers({ titles: list.titles, chunks: status.chunks });
          setActiveDoc((prev) => (prev && list.titles.includes(prev) ? prev : null));
        }
      })
      .catch(() => {
        if (!cancelled) setPapers(null);
      });

    return () => {
      cancelled = true;
    };
  }, [activeId, libraryId, sessions]);

  // auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, busy]);

  // auto-grow the composer textarea
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  function saveMsgs(sid: string, next: Msg[]) {
    lsWrite(msgsKey(sid), next);
  }

  function updateSessions(next: SessionMeta[]) {
    setSessions(next);
    if (email) lsWrite(sessionsKey(email), next);
  }

  function newSession() {
    if (!email) return;
    const sid = crypto.randomUUID();
    const meta: SessionMeta = {
      id: sid,
      name: "New Session",
      createdAt: new Date().toISOString(),
      isNamed: false,
    };
    setSessions((prev) => {
      const next = [meta, ...prev];
      if (email) lsWrite(sessionsKey(email), next);
      return next;
    });
    setActiveId(sid);
    setSidebarOpen(false);
  }

  async function signOut() {
    try {
      await authClient.signOut();
    } catch {
      // clear the user regardless — never trap them in the app
    }
    router.replace("/auth/sign-in");
    router.refresh();
  }

  async function deleteSession(sid: string) {
    try {
      await apiDeleteSessionVectors(sid);
    } catch {
      // vectors may not exist yet — ignore
    }
    const next = sessions.filter((s) => s.id !== sid);
    lsRemove(msgsKey(sid));
    if (next.length > 0) {
      updateSessions(next);
      setActiveId(next[0].id);
    } else {
      updateSessions([]);
      newSession();
    }
  }

  // ── send: main chat + /btw side channel ────────────────────────────────────
  async function send() {
    if (!input.trim() || busy || !activeId) return;
    const prompt = input.trim();
    setInput("");
    const isBtw = prompt.toLowerCase().startsWith("/btw");

    if (isBtw) {
      const query = prompt.slice(4).trim();
      setMsgs((m) => [...m, { role: "user", content: prompt }]);
      if (!query) {
        setMsgs((m) => [
          ...m,
          { role: "assistant", content: "Please add a question after `/btw`, e.g. `/btw What is attention?`", sideNote: true },
        ]);
        return;
      }
      setBusy(true);
      try {
        setMsgs((m) => [...m, { role: "assistant", content: "", sideNote: true }]);
        await apiBtw(query, (text) => {
          setMsgs((m) => {
            const next = [...m];
            next[next.length - 1] = { ...next[next.length - 1], content: next[next.length - 1].content + text };
            return next;
          });
        });
      } catch (e) {
        setMsgs((m) => [...m, { role: "assistant", content: `Error: ${(e as Error).message}` }]);
      } finally {
        setBusy(false);
      }
      setMsgs((m) => {
        saveMsgs(activeId, m);
        return m;
      });
      return;
    }

    // main chat
    setMsgs((m) => [...m, { role: "user", content: prompt }]);
    setMsgs((m) => [...m, { role: "assistant", content: "" }]);
    setBusy(true);
    const patchLast = (fn: (x: Msg) => Msg) =>
      setMsgs((m) => {
        const next = [...m];
        next[next.length - 1] = fn(next[next.length - 1]);
        return next;
      });
    try {
      const res = await apiChatStream(
        activeId,
        prompt,
        libraryScope(),
        activeDoc ? [activeDoc] : undefined,
        {
          onDelta: (text) => patchLast((x) => ({ ...x, content: x.content + text })),
          onReplace: (text) => patchLast((x) => ({ ...x, content: text })),
        },
      );
      patchLast((x) => ({ ...x, content: res.answer, route: res.route, sources: res.sources }));
      setMsgs((m) => {
        saveMsgs(activeId, m);
        return m;
      });

      const meta = sessions.find((s) => s.id === activeId);
      if (meta && !meta.isNamed) {
        try {
          const { name } = await apiRename(activeId, prompt);
          updateSessions(sessions.map((s) => (s.id === activeId ? { ...s, name, isNamed: true } : s)));
        } catch {
          // keep "New Session"
        }
      }
    } catch (e) {
      patchLast((x) => ({ ...x, content: `Error: ${(e as Error).message}` }));
      setMsgs((m) => {
        saveMsgs(activeId, m);
        return m;
      });
    } finally {
      setBusy(false);
    }
  }

  // ── document loaders (all target the per-user library) ──────────────────────
  function refreshPapers() {
    if (!libraryId) return;
    const scope = libraryScope();
    Promise.all([apiPapersList(libraryId, scope), apiPaperChunks(libraryId, scope)])
      .then(([list, status]) => setPapers({ titles: list.titles, chunks: status.chunks }))
      .catch(() => setPapers(null));
  }

  async function uploadFiles(files: File[]) {
    const target = libraryId ?? activeId;
    if (!target || files.length === 0) return;
    setBusy(true);
    try {
      for (const f of files) {
        try {
          const r = await apiUpload(target, f);
          addFlash("success", `Added: ${f.name} (${r.chunks} chunks indexed)`);
        } catch (e) {
          addFlash("error", `Failed: ${f.name} — ${(e as Error).message}`);
        }
      }
      refreshPapers();
    } finally {
      setBusy(false);
    }
  }

  function onFilesPicked(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (fileRef.current) fileRef.current.value = "";
    if (files.length === 0) {
      addFlash("info", "No files selected. Choose files first.");
      return;
    }
    void uploadFiles(files);
  }

  async function loadUrls() {
    const target = libraryId ?? activeId;
    if (!target) return;
    const urls = urlInput.split("\n").map((u) => u.trim()).filter(Boolean);
    if (urls.length === 0) {
      addFlash("info", "Enter at least one URL.");
      return;
    }
    setBusy(true);
    try {
      for (const u of urls) {
        try {
          const r = await apiWebpage(target, u);
          addFlash("success", `Loaded: ${u.slice(0, 60)} (${r.chunks} chunks)`);
        } catch (e) {
          addFlash("error", `Failed: ${u.slice(0, 60)} — ${(e as Error).message}`);
        }
      }
      setUrlInput("");
      refreshPapers();
    } finally {
      setBusy(false);
    }
  }

  async function loadArxiv() {
    const target = libraryId ?? activeId;
    if (!target) return;
    const q = arxivInput.trim();
    if (!q) {
      addFlash("info", "Enter a paper title or ArXiv ID.");
      return;
    }
    setBusy(true);
    try {
      const r = await apiArxiv(target, q);
      addFlash("success", `Loaded from ArXiv (${r.chunks} chunks)`);
      setArxivInput("");
      refreshPapers();
    } catch (e) {
      addFlash("error", `Failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  // ── render ─────────────────────────────────────────────────────────────────
  const activeSession = sessions.find((s) => s.id === activeId);
  const inputClass =
    "w-full rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-slate-200 placeholder:text-slate-500 outline-none transition focus:border-astra-400/50 focus:ring-2 focus:ring-astra-400/15";
  const actionClass =
    "w-full rounded-xl border border-white/10 bg-white/[0.05] px-3 py-2 text-xs font-medium text-slate-200 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <div className="relative flex h-dvh overflow-hidden">
      {sidebarOpen && (
        <button
          aria-label="Close sidebar"
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-20 bg-black/60 backdrop-blur-sm md:hidden"
        />
      )}

      {/* ── Sidebar ── */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 flex w-72 shrink-0 flex-col border-r border-white/[0.06] bg-[#060911]/85 backdrop-blur-2xl transition-transform duration-200 md:static md:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-2.5 px-3 pb-3 pt-4">
          <span className="grid h-8 w-8 place-items-center rounded-xl border border-astra-300/20 bg-gradient-to-br from-astra-400/25 to-astra-600/10 text-astra-200">
            <Sparkle className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium tracking-wide text-slate-100">Papeer</div>
            <div className="truncate text-[10px] text-slate-500">{email ?? "Not signed in"}</div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-white/5 hover:text-slate-200 md:hidden"
            aria-label="Close sidebar"
          >
            <Close />
          </button>
        </div>

        <div className="px-3 pb-3">
          <button
            onClick={newSession}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-astra-300/20 bg-astra-400/10 px-3 py-2 text-xs font-medium text-astra-100 transition hover:border-astra-300/30 hover:bg-astra-400/15"
          >
            <Plus className="h-3.5 w-3.5" /> New chat
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-5">
          <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            Sessions
          </div>
          <div className="mb-5 space-y-0.5">
            {sessions.length === 0 && (
              <div className="rounded-lg px-2 py-2 text-[11px] text-slate-600">No sessions yet.</div>
            )}
            {[...sessions]
              .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
              .map((s) => (
                <div key={s.id} className="group flex items-center">
                  <button
                    onClick={() => {
                      setActiveId(s.id);
                      setSidebarOpen(false);
                    }}
                    className={`flex-1 truncate rounded-lg px-2.5 py-1.5 text-left text-xs transition ${
                      s.id === activeId
                        ? "bg-white/[0.08] text-white"
                        : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200"
                    }`}
                    title={s.name}
                  >
                    {s.name}
                  </button>
                  <button
                    onClick={() => deleteSession(s.id)}
                    className="rounded-lg p-1.5 text-slate-600 opacity-0 transition hover:bg-white/5 hover:text-red-300 group-hover:opacity-100 focus:opacity-100"
                    title="Delete session"
                    aria-label={`Delete ${s.name}`}
                  >
                    <Trash />
                  </button>
                </div>
              ))}
          </div>

          <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            Documents
          </div>

          <div className="space-y-4">
            <div>
              <div className="mb-1 text-[11px] font-medium text-slate-300">Files (PDF/TXT/MD)</div>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".pdf,.txt,.md,.markdown"
                onChange={onFilesPicked}
                className="hidden"
              />
              <button onClick={() => fileRef.current?.click()} disabled={busy} className={actionClass}>
                {busy ? "Uploading…" : "Add files (PDF/TXT/MD)"}
              </button>
            </div>

            <div>
              <div className="mb-1 text-[11px] font-medium text-slate-300">Web pages</div>
              <textarea
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="https://example.com/paper (one per line)"
                rows={2}
                className={`${inputClass} resize-none`}
              />
              <button onClick={loadUrls} disabled={busy} className={`${actionClass} mt-1.5`}>
                Load URLs
              </button>
            </div>

            <div>
              <div className="mb-1 text-[11px] font-medium text-slate-300">ArXiv papers</div>
              <input
                value={arxivInput}
                onChange={(e) => setArxivInput(e.target.value)}
                placeholder="1706.03762 or Attention Is All You Need"
                className={inputClass}
              />
              <button onClick={loadArxiv} disabled={busy} className={`${actionClass} mt-1.5`}>
                Load ArXiv paper
              </button>
            </div>
          </div>

          {flash.length > 0 && (
            <div className="mt-4 space-y-1">
              {flash.map((f, i) => (
                <div
                  key={i}
                  className={`rounded-lg border px-2 py-1.5 text-[10px] leading-snug ${
                    f.kind === "error"
                      ? "border-red-400/20 bg-red-400/10 text-red-200"
                      : f.kind === "success"
                        ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-200"
                        : "border-amber-400/20 bg-amber-400/10 text-amber-200"
                  }`}
                >
                  {f.text}
                </div>
              ))}
            </div>
          )}

          <div className="mt-4 border-t border-white/[0.06] pt-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                Loaded
              </span>
              {papers && papers.chunks > 0 && (
                <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[9px] text-slate-400">
                  {papers.chunks} chunks
                </span>
              )}
            </div>
            {papers === null ? (
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] text-slate-600">Could not load document list.</span>
                <button
                  onClick={refreshPapers}
                  className="shrink-0 rounded-lg border border-white/10 bg-white/[0.05] px-2 py-1 text-[10px] text-slate-300 transition hover:bg-white/10"
                >
                  Retry
                </button>
              </div>
            ) : papers.titles.length === 0 ? (
              <div className="text-[10px] text-slate-600">No documents loaded yet.</div>
            ) : (
              <div className="space-y-1">
                <div className="mb-1 text-[9px] text-slate-600">
                  {activeDoc ? "Focused — click again to search all documents" : "Click a document to focus answers on it"}
                </div>
                {papers.titles.map((t) => {
                  const selected = t === activeDoc;
                  return (
                    <button
                      key={t}
                      onClick={() => setActiveDoc(selected ? null : t)}
                      disabled={busy}
                      title={t}
                      className={`flex w-full items-center gap-1.5 rounded-lg px-1.5 py-1 text-left text-[10px] transition ${
                        selected
                          ? "border border-astra-300/30 bg-astra-400/15 text-astra-100"
                          : "border border-transparent text-slate-400 hover:bg-white/[0.04] hover:text-slate-200"
                      }`}
                    >
                      <Doc className={`h-3 w-3 shrink-0 ${selected ? "text-astra-200" : "text-astra-300/70"}`} />
                      <span className="truncate">{t}</span>
                      {selected && <span className="ml-auto shrink-0 text-[9px] text-astra-200">focused</span>}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-white/[0.06] p-3">
          <button
            onClick={signOut}
            className="flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-xs text-slate-400 transition hover:bg-white/5 hover:text-red-200"
          >
            <Logout />
            Sign out
          </button>
        </div>
      </aside>

      {/* ── Chat ── */}
      <main className="relative flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-white/[0.06] px-4 py-3 md:px-6">
          <button
            onClick={() => setSidebarOpen(true)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-white/5 hover:text-slate-200 md:hidden"
            aria-label="Open sidebar"
          >
            <Menu />
          </button>
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-slate-100">
              {activeSession?.name ?? "New Session"}
            </div>
            <div className="text-[10px] text-slate-500">Grounded answers with sources</div>
          </div>
          <div className="ml-auto hidden items-center gap-2 sm:flex">
            <span className="max-w-[220px] truncate rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[10px] text-slate-400">
              {email ?? "Not signed in"}
            </span>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-4 py-8 md:px-6">
          <div className="mx-auto w-full max-w-5xl space-y-6">
            {msgs.length === 0 && (
              <div className="mt-6 rounded-3xl border border-white/[0.06] bg-white/[0.02] p-8 text-center backdrop-blur-sm md:p-10">
                <span className="mx-auto mb-5 grid h-12 w-12 place-items-center rounded-2xl border border-astra-300/20 bg-gradient-to-br from-astra-400/25 to-astra-600/10 text-astra-200 shadow-[0_0_40px_rgba(91,125,255,0.2)]">
                  <Sparkle className="h-5 w-5" />
                </span>
                <h2 className="text-lg font-light tracking-tight text-white md:text-xl">
                  Ask your papers anything.
                </h2>
                <p className="mx-auto mt-2 max-w-md text-xs leading-relaxed text-slate-400 md:text-sm">
                  Upload documents in the sidebar, then ask questions, verify claims or search the web.
                </p>
                <div className="mt-5 flex flex-wrap items-center justify-center gap-2 text-[10px] text-slate-500">
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1">Vector retrieval</span>
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1">Web search</span>
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1">Claim verification</span>
                </div>
              </div>
            )}

            {msgs.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`min-w-0 ${m.role === "user" ? "max-w-[85%]" : "w-full max-w-[92%]"}`}>
                  {m.role === "assistant" && (
                    <div className="mb-1.5 flex items-center gap-1.5 text-[9px] font-medium uppercase tracking-[0.2em] text-slate-500">
                      <span className="h-1 w-1 rounded-full bg-astra-400" />
                      {m.sideNote ? "Side note" : "Papeer"}
                    </div>
                  )}
                  <div
                    className={`overflow-x-auto px-4 py-3 text-sm ${
                      m.role === "user"
                        ? "rounded-2xl rounded-br-md bg-gradient-to-br from-astra-500 to-[#6d5bff] text-white shadow-lg shadow-astra-600/20"
                        : m.sideNote
                          ? "rounded-2xl rounded-bl-md border border-dashed border-astra-300/20 bg-white/[0.02] italic text-slate-300"
                          : "glass rounded-2xl rounded-bl-md text-slate-200 shadow-xl shadow-black/20"
                    }`}
                  >
                    {m.role === "user" ? (
                      m.content
                    ) : m.content ? (
                      <Markdown text={m.content} />
                    ) : null}
                    {m.sideNote && m.content && (
                      <div className="mt-1.5 text-[10px] font-normal not-italic text-slate-500">
                        Not used for paper Q&amp;A context
                      </div>
                    )}
                  </div>

                  {m.role === "assistant" && !m.sideNote && m.route && (
                    <div className="mt-2">
                      <RouteBadge route={m.route} />
                    </div>
                  )}

                  {m.role === "assistant" && !m.sideNote && m.sources && m.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] text-slate-500">
                      <span className="uppercase tracking-[0.14em]">Sources</span>
                      {m.sources.map((s, si) =>
                        s.image_url ? (
                          <a
                            key={si}
                            href={s.image_url}
                            target="_blank"
                            rel="noreferrer"
                            className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] py-0.5 pl-0.5 pr-2 text-slate-300 transition hover:border-astra-300/30 hover:bg-white/10"
                            title={s.title}
                          >
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={s.image_url}
                              alt={`Figure ${s.figure ?? ""} from ${s.title}`}
                              className="h-6 w-6 rounded-full object-cover"
                            />
                            <span className="max-w-[14rem] truncate">
                              {s.title}
                              {s.figure != null ? ` · Fig.${s.figure}` : ""}
                              {s.page != null ? ` · p.${s.page}` : ""}
                            </span>
                          </a>
                        ) : s.url ? (
                          <a
                            key={si}
                            href={s.url}
                            target="_blank"
                            rel="noreferrer"
                            className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-slate-300 transition hover:border-astra-300/30 hover:bg-white/10"
                          >
                            {s.title}
                          </a>
                        ) : (
                          <span
                            key={si}
                            className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-slate-300"
                          >
                            {s.title}
                            {s.page != null ? ` · p.${s.page}` : ""}
                          </span>
                        ),
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {busy && (
              <div className="flex justify-start">
                <div className="glass flex items-center gap-2 rounded-2xl rounded-bl-md px-4 py-3 text-xs text-slate-400">
                  <span className="flex gap-1">
                    <span className="h-1.5 w-1.5 rounded-full bg-astra-300" style={{ animation: "pulse-soft 1.2s ease-in-out infinite" }} />
                    <span className="h-1.5 w-1.5 rounded-full bg-astra-300" style={{ animation: "pulse-soft 1.2s ease-in-out 0.2s infinite" }} />
                    <span className="h-1.5 w-1.5 rounded-full bg-astra-300" style={{ animation: "pulse-soft 1.2s ease-in-out 0.4s infinite" }} />
                  </span>
                  Thinking…
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* ── Composer ── */}
        <div className="px-4 pb-5 pt-2 md:px-6">
          <div className="mx-auto w-full max-w-5xl">
            <div className="glass-strong rounded-[1.5rem] p-2 shadow-2xl shadow-black/50">
              <textarea
                ref={taRef}
                value={input}
                rows={1}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                placeholder="Ask about your papers, verify a claim, or search the web…"
                className="max-h-40 w-full resize-none overflow-y-auto bg-transparent px-3 py-2 text-sm leading-relaxed text-slate-100 outline-none placeholder:text-slate-500"
              />
              <div className="flex items-center gap-2 px-1 pb-1">
                <button
                  onClick={() => quickFileRef.current?.click()}
                  disabled={busy || !activeId}
                  className="grid h-7 w-7 place-items-center rounded-full border border-white/10 bg-white/[0.04] text-slate-300 transition hover:bg-white/10 disabled:opacity-40"
                  title="Upload a document"
                  aria-label="Upload a document"
                >
                  <Plus className="h-3.5 w-3.5" />
                </button>
                  <span
                    className={`flex max-w-[45%] items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] ${
                      activeDoc
                        ? "border-astra-300/30 bg-astra-400/15 text-astra-100"
                        : "border-white/10 bg-white/[0.04] text-slate-300"
                    }`}
                    title={activeDoc ? `Retrieving only from: ${activeDoc}` : "Retrieving from all loaded documents"}
                  >
                    <Sparkle className={`h-3 ${activeDoc ? "text-astra-200" : "text-astra-300"}`} />
                    {activeDoc ? (
                      <span className="truncate">Focused: {activeDoc}</span>
                    ) : (
                      "Papeer RAG"
                    )}
                  </span>
                <button
                  onClick={send}
                  disabled={busy || !input.trim()}
                  className="ml-auto grid h-8 w-8 place-items-center rounded-full bg-white text-[#05070d] transition hover:bg-astra-100 disabled:cursor-not-allowed disabled:bg-white/20 disabled:text-slate-500"
                  aria-label="Send message"
                >
                  <ArrowUp className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
            <div className="mt-2 text-center text-[10px] text-slate-600">
              Enter to send · Shift+Enter for a new line · <span className="text-slate-500">/btw</span> for a side note
            </div>
            <input
              ref={quickFileRef}
              type="file"
              multiple
              accept=".pdf,.txt,.md,.markdown"
              className="hidden"
              onChange={(e) => {
                const files = Array.from(e.target.files ?? []);
                e.currentTarget.value = "";
                if (files.length > 0) void uploadFiles(files);
              }}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
