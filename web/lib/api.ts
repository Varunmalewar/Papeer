"use client";

/**
 * Frontend API client.
 * Saare requests Next.js API routes (/api/chat, /api/upload, /api/papeer/*)
 * se hote hain. Wahan se server-side PAPEER_API_KEY ke saath FastAPI backend
 * pe jaate hain. Browser ko koi API key expose nahi hoti, CORS issue bhi nahi.
 */

export type Source = { title: string; page?: number; url?: string; figure?: number; image_url?: string };
export type HistoryMsg = { role: "user" | "assistant"; content: string };

async function parseError(res: Response, fallback: string): Promise<Error> {
  const data = await res.json().catch(() => ({}));
  const msg = (data && typeof data === "object" && (data.error || data.detail)) || fallback;
  return new Error(String(msg));
}

export async function apiChat(
  sessionId: string,
  query: string,
  docScopeIds?: string[],
  docTitles?: string[],
) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      query,
      doc_scope_ids: docScopeIds,
      doc_titles: docTitles && docTitles.length > 0 ? docTitles : undefined,
    }),
  });
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, `Chat failed (${res.status}).`);
  return res.json() as Promise<{
    answer: string;
    route: string | null;
    sources: Source[];
  }>;
}

export async function apiUpload(sessionId: string, file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`/api/upload?session_id=${sessionId}`, {
    method: "POST",
    body: form,
  });
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, `Upload failed (${res.status}).`);
  return res.json() as Promise<JobStart>;
}

/** Uploads and ArXiv loads return a job at once; poll it until it is done or failed. */
export type JobStart = { job_id: string; status: string };

export async function apiJob(jobId: string) {
  const res = await fetch(`/api/papeer/papers/jobs/${jobId}`);
  if (res.status === 401) throw new Error("Login required.");
  if (res.status === 404) throw new Error("Status was lost (server restarted). Refresh the document list.");
  if (!res.ok) throw await parseError(res, `Status failed (${res.status}).`);
  return res.json() as Promise<{ state: "processing" | "done" | "error"; chunks: number | null; error: string | null }>;
}

/** Chat history restore from the LangGraph checkpointer. */
export async function apiHistory(sessionId: string) {
  const res = await fetch(`/api/papeer/chat/${sessionId}/history`);
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, `History failed (${res.status}).`);
  return res.json() as Promise<{ messages: HistoryMsg[] }>;
}

/** LLM-generated 3-5 word session title (first message). */
export async function apiRename(sessionId: string, query: string) {
  const res = await fetch("/api/papeer/chat/rename", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, query }),
  });
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, "Rename failed.");
  return res.json() as Promise<{ name: string }>;
}

/** /btw side channel — streams plain text; onChunk fires per chunk. */
export async function apiBtw(query: string, onChunk: (text: string) => void) {
  const res = await fetch("/api/papeer/btw", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, `btw failed (${res.status}).`);
  const reader = res.body?.getReader();
  if (!reader) throw new Error("Streaming not supported.");
  const decoder = new TextDecoder();
  let full = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value, { stream: true });
    full += text;
    onChunk(text);
  }
  return full;
}

export async function apiChatStream(
  sessionId: string,
  query: string,
  docScopeIds: string[] | undefined,
  docTitles: string[] | undefined,
  handlers: { onDelta?: (text: string) => void; onReplace?: (text: string) => void },
) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      query,
      doc_scope_ids: docScopeIds,
      doc_titles: docTitles && docTitles.length > 0 ? docTitles : undefined,
    }),
  });
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, `Chat failed (${res.status}).`);
  const reader = res.body?.getReader();
  if (!reader) throw new Error("Streaming not supported.");
  const decoder = new TextDecoder();
  let buffer = "";
  let done: { answer: string; route: string | null; sources: Source[] } | null = null;
  for (;;) {
    const { done: finished, value } = await reader.read();
    if (finished) break;
    buffer += decoder.decode(value, { stream: true });
    let split = buffer.indexOf("\n\n");
    while (split >= 0) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
      if (dataLine) {
        let event: Record<string, unknown> | null = null;
        try {
          event = JSON.parse(dataLine.slice(5).trim());
        } catch {
          event = null;
        }
        if (event) {
          if (event.type === "delta" && typeof event.text === "string") handlers.onDelta?.(event.text);
          else if (event.type === "replace" && typeof event.text === "string") handlers.onReplace?.(event.text);
          else if (event.type === "error") throw new Error(String(event.error || "Chat failed."));
          else if (event.type === "done") {
            done = {
              answer: String(event.answer ?? ""),
              route: (event.route as string | null) ?? null,
              sources: (event.sources as Source[]) ?? [],
            };
          }
        }
      }
      split = buffer.indexOf("\n\n");
    }
  }
  if (!done) throw new Error("Stream ended without a response.");
  return done;
}

function scopeQuery(scopeIds?: string[]) {
  const ids = (scopeIds ?? []).filter(Boolean);
  return ids.length ? `?scope_ids=${encodeURIComponent(ids.join(","))}` : "";
}

export async function apiPapersList(sessionId: string, scopeIds?: string[]) {
  const res = await fetch(`/api/papeer/papers/${sessionId}${scopeQuery(scopeIds)}`);
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, "Could not load documents.");
  return res.json() as Promise<{ titles: string[] }>;
}

export async function apiPaperChunks(sessionId: string, scopeIds?: string[]) {
  const res = await fetch(`/api/papeer/papers/${sessionId}/status${scopeQuery(scopeIds)}`);
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, "Could not load vector store status.");
  return res.json() as Promise<{ chunks: number }>;
}

export async function apiArxiv(sessionId: string, query: string) {
  const res = await fetch(
    `/api/papeer/papers/arxiv?session_id=${encodeURIComponent(sessionId)}&query=${encodeURIComponent(query)}`,
    { method: "POST" },
  );
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, "ArXiv load failed.");
  return res.json() as Promise<JobStart>;
}

export async function apiWebpage(sessionId: string, url: string) {
  const res = await fetch(
    `/api/papeer/papers/webpage?session_id=${encodeURIComponent(sessionId)}&url=${encodeURIComponent(url)}`,
    { method: "POST" },
  );
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, "Web page load failed.");
  return res.json() as Promise<{ chunks: number }>;
}

export async function apiDeleteSessionVectors(sessionId: string) {
  const res = await fetch(`/api/papeer/papers/${sessionId}`, { method: "DELETE" });
  if (res.status === 401) throw new Error("Login required.");
  if (!res.ok) throw await parseError(res, "Delete failed.");
  return res.json() as Promise<{ deleted_chunks: number }>;
}
