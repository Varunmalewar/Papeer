"use client";

import { Suspense } from "react";
import { NeonAuthUIProvider, AuthView } from "@neondatabase/auth-ui";
import { usePathname } from "next/navigation";
import { authClient } from "@/lib/auth/client";
import { AuthShell } from "@/components/auth-shell";
import "@neondatabase/auth-ui/css";

function AuthViewByPath() {
  const pathname = usePathname();
  return <AuthView pathname={pathname} redirectTo="/chat" />;
}

export default function AuthCatchAllPage() {
  return (
    <NeonAuthUIProvider authClient={authClient} emailVerification={{ otp: true }} defaultTheme="dark">
      <AuthShell title="Authentication" subtitle="Sign in to continue to Papeer.">
        <Suspense fallback={<div className="h-24 animate-pulse rounded-xl bg-white/5" />}>
          <AuthViewByPath />
        </Suspense>
      </AuthShell>
    </NeonAuthUIProvider>
  );
}
