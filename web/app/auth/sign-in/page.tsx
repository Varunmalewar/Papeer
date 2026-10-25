"use client";
import { NeonAuthUIProvider, AuthView } from "@neondatabase/auth-ui";
import { usePathname } from "next/navigation";
import { authClient } from "@/lib/auth/client";
import { AuthShell } from "@/components/auth-shell";
import "@neondatabase/auth-ui/css";

export default function SignInPage() {
  const pathname = usePathname();
  return (
    <NeonAuthUIProvider authClient={authClient} emailVerification={{ otp: true }} defaultTheme="dark">
      <AuthShell title="Welcome back" subtitle="Sign in to continue your research.">
        <AuthView pathname={pathname} redirectTo="/chat" />
      </AuthShell>
    </NeonAuthUIProvider>
  );
}
