import { NextRequest, NextResponse } from "next/server";
import { getTrustedUser } from "@/lib/user-scope";

/**
 * Generic catch-all proxy to the FastAPI backend.
 * forwards /api/papeer/<path> -> BACKEND/<path> with the server-side API key.
 * text/plain responses (e.g. /btw stream) are passed through untouched.
 *
 * Guardrail: the Neon Auth session is verified server-side and the derived
 * library id is forwarded as X-Papeer-User. Document reads/writes are forced
 * onto the user's own library — GET /papers/<sid> + ?scope_ids=... are
 * rewritten to the trusted library id (client-supplied scope is dropped).
 */
const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.PAPEER_API_KEY ?? "";

type Ctx = { params: Promise<{ path: string[] }> };

async function proxy(req: NextRequest, ctx: Ctx) {
  try {
    const user = await getTrustedUser();
    if (!user) {
      return NextResponse.json({ error: "Login required." }, { status: 401 });
    }
    const { path } = await ctx.params;
    const method = req.method;
    const url = new URL(req.url);
    const params = new URLSearchParams(url.searchParams);
    const segs = [...path];

    if (segs[0] === "papers") {
      const seg1 = segs[1] ?? "";
      if (method === "POST" && ["upload", "arxiv", "webpage"].includes(seg1)) {
        // uploads always target the user's own library
        params.set("session_id", user);
      } else if (method === "GET" && seg1 && seg1 !== "sessions" && segs[2] !== undefined && segs[2] === "status") {
        // GET /papers/<sid>/status — force the trusted library
        segs[1] = user;
        params.delete("scope_ids");
      } else if (method === "GET" && seg1 && segs[2] === undefined && !["upload", "arxiv", "webpage", "sessions"].includes(seg1)) {
        // GET /papers/<sid> — document list from the trusted library
        segs[1] = user;
        params.delete("scope_ids");
      }
    }

    const target = `${BACKEND}/${segs.join("/")}${params.size ? `?${params}` : ""}`;
    const headers = new Headers({ "X-API-Key": API_KEY, "X-Papeer-User": user });
    const ct = req.headers.get("content-type");
    if (ct) headers.set("Content-Type", ct);
    const body =
      method === "GET" || method === "DELETE" || method === "HEAD"
        ? undefined
        : await req.arrayBuffer();
    const res = await fetch(target, { method, headers, body });
    if ((res.headers.get("content-type") ?? "").includes("text/plain")) {
      return new Response(res.body, {
        status: res.status,
        headers: { "Content-Type": "text/plain" },
      });
    }
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text || `Backend returned status ${res.status}` };
    }
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function GET(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx);
}
export async function POST(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx);
}
export async function DELETE(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx);
}
export async function PUT(req: NextRequest, ctx: Ctx) {
  return proxy(req, ctx);
}
