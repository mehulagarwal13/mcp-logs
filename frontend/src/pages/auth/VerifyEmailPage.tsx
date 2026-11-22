import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { CheckCircle2, MailWarning, XCircle } from "lucide-react";
import { verifyEmail } from "@/api/auth";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { ResendVerification } from "./ResendVerification";
import type { ApiError } from "@/types/common";

type Phase =
  | { kind: "verifying" }
  | { kind: "success" }
  | { kind: "missing" }
  | { kind: "invalid" }
  | { kind: "expired" }
  | { kind: "used" }
  | { kind: "conflict"; message: string }
  | { kind: "error" };

/**
 * Reads the verification token from the URL FRAGMENT (`/verify-email#token=...`).
 * Fragments are never sent to servers or in `Referer`, so the token stays out
 * of access logs and proxies; it is removed from the address bar immediately
 * and kept in memory only -- never in localStorage/sessionStorage, never logged.
 */
function takeTokenFromUrl(): string | null {
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const token = params.get("token");
  if (window.location.hash) {
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  return token && token.length > 0 ? token : null;
}

/**
 * `/verify-email`. The success state is reached ONLY when `POST
 * /auth/verify-email` resolved -- the server consuming the token is the proof,
 * never anything held client-side. A successful verification returns no
 * session; the user signs in through the normal login page.
 */
export function VerifyEmailPage() {
  const [phase, setPhase] = useState<Phase>({ kind: "verifying" });
  const [resendEmail, setResendEmail] = useState("");
  // The token can only be consumed once. React StrictMode re-runs effects in
  // development; this guard (refs survive that re-run) makes sure the single
  // request is not sent twice, which would turn a success into "already used".
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    const token = takeTokenFromUrl();
    if (!token) {
      setPhase({ kind: "missing" });
      return;
    }
    verifyEmail(token)
      .then(() => setPhase({ kind: "success" }))
      .catch((err: unknown) => {
        const apiError = err as ApiError;
        switch (apiError?.errorCode) {
          case "email_verification.expired":
            setPhase({ kind: "expired" });
            break;
          case "email_verification.already_used":
            setPhase({ kind: "used" });
            break;
          case "email_verification.invalid":
            setPhase({ kind: "invalid" });
            break;
          case "organization.slug_taken":
          case "auth.email_taken":
          case "invitation.not_pending":
          case "invitation.expired":
          case "invitation.not_found":
            // The link was fine but activation could not complete (and the
            // server rolled back, so nothing was consumed or created).
            setPhase({ kind: "conflict", message: apiError.message });
            break;
          default:
            setPhase({ kind: "error" });
        }
      });
  }, []);

  if (phase.kind === "verifying") {
    return (
      <div className="glass-panel rounded-2xl px-6 py-6 text-center" role="status" aria-live="polite">
        <h1 className="mb-2 text-lg font-semibold text-ink">Confirming your email…</h1>
        <p className="text-sm text-ink-muted">One moment.</p>
      </div>
    );
  }

  if (phase.kind === "success") {
    return (
      <div className="glass-panel rounded-2xl px-6 py-6 text-center">
        <div className="mb-3 flex justify-center text-success">
          <CheckCircle2 className="h-9 w-9" />
        </div>
        <h1 className="mb-2 text-lg font-semibold text-ink">Email confirmed</h1>
        <p className="mb-5 text-sm text-ink-muted">
          Thanks — your email address is verified. Sign in to continue.
        </p>
        <Link to="/login">
          <Button variant="primary" className="w-full">
            Go to sign in
          </Button>
        </Link>
      </div>
    );
  }

  if (phase.kind === "used") {
    return (
      <div className="glass-panel rounded-2xl px-6 py-6 text-center">
        <h1 className="mb-2 text-lg font-semibold text-ink">This link was already used</h1>
        <p className="mb-5 text-sm text-ink-muted">
          If you already confirmed your email, you can sign in. Otherwise request a new link below.
        </p>
        <Link to="/login">
          <Button variant="primary" className="mb-3 w-full">
            Go to sign in
          </Button>
        </Link>
        <ResendForm email={resendEmail} onEmailChange={setResendEmail} />
      </div>
    );
  }

  if (phase.kind === "conflict") {
    return (
      <div className="glass-panel rounded-2xl px-6 py-6 text-center">
        <div className="mb-3 flex justify-center text-warning">
          <MailWarning className="h-9 w-9" />
        </div>
        <h1 className="mb-2 text-lg font-semibold text-ink">We couldn&apos;t finish confirming</h1>
        <p className="mb-5 text-sm text-ink-muted">{phase.message}</p>
        <Link to="/signup" className="text-sm font-medium text-accent hover:underline">
          Start again
        </Link>
      </div>
    );
  }

  const copy = {
    missing: {
      title: "This confirmation link is incomplete",
      body: "Open the full link from your email, or request a new one below.",
    },
    invalid: {
      title: "This confirmation link isn’t valid",
      body: "It may have been replaced by a newer email. Use the most recent one, or request a new link below.",
    },
    expired: {
      title: "This confirmation link has expired",
      body: "Links are valid for a day. Request a new one below.",
    },
    error: {
      title: "Something went wrong",
      body: "We couldn’t confirm your email just now. Please try the link again in a moment, or request a new one.",
    },
  }[phase.kind];

  return (
    <div className="glass-panel rounded-2xl px-6 py-6 text-center">
      <div className="mb-3 flex justify-center text-critical">
        <XCircle className="h-9 w-9" />
      </div>
      <h1 className="mb-2 text-lg font-semibold text-ink">{copy.title}</h1>
      <p className="mb-5 text-sm text-ink-muted">{copy.body}</p>
      <ResendForm email={resendEmail} onEmailChange={setResendEmail} />
      <p className="mt-6 text-xs text-ink-subtle">
        <Link to="/login" className="font-medium text-accent hover:underline">
          Back to sign in
        </Link>
      </p>
    </div>
  );
}

function ResendForm({
  email,
  onEmailChange,
}: {
  email: string;
  onEmailChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-col gap-2 text-left">
      <label htmlFor="resend-email" className="text-xs font-medium text-ink-muted">
        Email address
      </label>
      <Input
        id="resend-email"
        type="email"
        placeholder="you@company.com"
        value={email}
        onChange={(e) => onEmailChange(e.target.value)}
      />
      <ResendVerification email={email.trim()} />
    </div>
  );
}
