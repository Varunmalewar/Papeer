"use client";
import { createAuthClient } from "@neondatabase/auth/next";

// No URL → baseURL defaults to same-origin /api/auth, which is proxied to
// Neon Auth by app/api/auth/[...path]/route.ts. Cookies stay on this origin,
// so the server-side session (lib/auth/server.ts) actually works.
export const authClient = createAuthClient();
