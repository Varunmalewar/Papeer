import { auth } from "@/lib/auth/server";

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - auth (auth pages and API routes)
     * - the root landing page (public marketing page)
     */
    "/((?!api|_next/static|_next/image|favicon.ico|auth|$).*)",
  ],
};

export default auth.middleware();
