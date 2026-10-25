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
    // Guardrail: uploads always land in the verified user's library.
    const target = `${BACKEND}/papers/upload?session_id=${encodeURIComponent(user)}`;
    const form = await req.formData();
    const res = await fetch(target, {
      method: "POST",
      headers: {
        "X-API-Key": API_KEY,
        "X-Papeer-User": user,
      },
      body: form,
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
