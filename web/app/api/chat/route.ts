import { NextRequest, NextResponse } from "next/server";
import { getTrustedUser } from "@/lib/user-scope";

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.PAPEER_API_KEY ?? "";

export async function POST(req: NextRequest) {
  try {
    const user = await getTrustedUser();
    if (!user) {
      return NextResponse.json({ error: "Login required." }, { status: 401 });
    }
    const body = await req.json();
    // Guardrail: retrieval scope is forced to the verified user's library —
    // client-supplied doc_scope_ids are ignored. doc_titles (document focus)
    // stays client-selectable but is user-filtered backend-side.
    body.doc_scope_ids = [user];
    const res = await fetch(`${BACKEND}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
        "X-Papeer-User": user,
      },
      body: JSON.stringify(body),
    });
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
