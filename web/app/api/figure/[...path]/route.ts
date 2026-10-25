import { NextRequest, NextResponse } from "next/server";
import { getTrustedUser } from "@/lib/user-scope";

/**
 * Binary figure proxy: /api/figure/<session>/<figureId> -> backend
 * /papers/<trustedLibrary>/figures/<figureId>.
 *
 * The catch-all /api/papeer proxy JSON-parses responses, so images need their
 * own route. Session segment is ignored on purpose — the trusted Neon Auth
 * library always wins, matching the ownership check in the backend endpoint.
 */
const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.PAPEER_API_KEY ?? "";

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(_req: NextRequest, ctx: Ctx) {
  try {
    const user = await getTrustedUser();
    if (!user) {
      return NextResponse.json({ error: "Login required." }, { status: 401 });
    }
    const { path } = await ctx.params;
    const figureId = path[path.length - 1];
    if (!figureId) {
      return NextResponse.json({ error: "Figure not found." }, { status: 404 });
    }
    const target = `${BACKEND}/papers/${encodeURIComponent(user)}/figures/${encodeURIComponent(figureId)}`;
    const res = await fetch(target, {
      headers: { "X-API-Key": API_KEY, "X-Papeer-User": user },
      cache: "no-store",
    });
    if (!res.ok || !res.body) {
      return NextResponse.json({ error: "Figure not found." }, { status: res.status || 404 });
    }
    return new Response(res.body, {
      status: 200,
      headers: {
        "Content-Type": res.headers.get("content-type") ?? "image/png",
        "Cache-Control": res.headers.get("cache-control") ?? "private, max-age=3600",
      },
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
