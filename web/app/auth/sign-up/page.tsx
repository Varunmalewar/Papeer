"use client";
import { NeonAuthUIProvider, AuthView } from "@neondatabase/auth-ui";
import { usePathname } from "next/navigation";
import { authClient } from "@/lib/auth/client";
import { AuthShell } from "@/components/auth-shell";
import "@neondatabase/auth-ui/css";

export default function SignUpPage() {
  const pathname = usePathname();
  return (
    <NeonAuthUIProvider authClient={authClient} emailVerification={{ otp: true }} defaultTheme="dark">
      <AuthShell title="Create your account" subtitle="Start reading papers with an assistant that cites its sources.">
        <AuthView pathname={pathname} redirectTo="/chat" />
      </AuthShell>
    </NeonAuthUIProvider>
  );
}
