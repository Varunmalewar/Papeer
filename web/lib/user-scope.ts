import { createHash } from "crypto";
import { auth } from "@/lib/auth/server";

/**
 * Server-side trusted user scope. Verifies the browser's Neon Auth session
 * (cookie-bound) and derives the SAME library id the client would derive
 * (lib-<sha256(email)[:32]>). Anything the client sends (session ids, scope
 * ids) is advisory only — the backend enforces THIS value on every vector
 * read/write via the X-Papeer-User header.
 */
export async function getTrustedUser(): Promise<string | null> {
  try {
    const { data } = await auth.getSession();
    const email = (data?.user?.email ?? "").trim().toLowerCase();
    if (!email) return null;
    return "lib-" + createHash("sha256").update(email).digest("hex").slice(0, 32);
  } catch {
    return null;
  }
}
