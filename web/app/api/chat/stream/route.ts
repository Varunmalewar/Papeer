import { NextRequest, NextResponse } from "next/server";
import { getTrustedUser } from "@/lib/user-scope";

/**
 * SSE chat proxy: /api/chat/stream -> backend /chat/stream.
 *
 * The generic /api/papeer proxy JSON-parses responses, so token streams need
 * their own pass-through route. Retrieval scope is forced to the verified
 * user's library; client-supplied doc_scope_ids are dropped.
 */
const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.PAPEER_API_KEY ?? "";

export async function POST(req: NextRequest) {
  try {
    const user = await getTrustedUser();
    if (!user) {
      return NextResponse.json({ error: "Login required." }, { status: 401 });
    }
    const body = await req.json();
    body.doc_scope_ids = [user];
    const res = await fetch(`${BACKEND}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
        "X-Papeer-User": user,
      },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        data = { error: text || `Backend returned status ${res.status}` };
      }
      return NextResponse.json(data, { status: res.status });
    }
    return new Response(res.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
