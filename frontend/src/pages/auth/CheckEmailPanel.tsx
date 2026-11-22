import { Mail } from "lucide-react";
import { Link } from "react-router-dom";
import { ResendVerification } from "./ResendVerification";

interface CheckEmailPanelProps {
  /** Address the user typed. Shown back to them; may be omitted when unknown. */
  email?: string;
  /** Short phrase for what the link completes, e.g. "create your account". */
  purpose: string;
}

/**
 * Pending-verification screen. Deliberately says nothing that depends on
 * whether the address already has an account (the backend answers signup
 * identically either way) and never implies the account already exists.
 */
export function CheckEmailPanel({ email, purpose }: CheckEmailPanelProps) {
  return (
    <div className="glass-panel rounded-2xl px-6 py-6">
      <div className="mb-3 flex justify-center">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-accent/10 text-accent">
          <Mail className="h-5 w-5" />
        </div>
      </div>
      <h1 className="mb-2 text-center text-lg font-semibold text-ink">Check your email</h1>
      <p className="mb-1 text-center text-sm text-ink-muted">
        {email ? (
          <>
            If <span className="font-medium text-ink">{email}</span> can be used, we&apos;ve sent a
            confirmation link.
          </>
        ) : (
          <>We&apos;ve sent a confirmation link to the invited email address.</>
        )}
      </p>
      <p className="mb-5 text-center text-sm text-ink-muted">
        Open it to {purpose}. The link works once and expires after a day. Nothing is created until
        you do.
      </p>
      {email ? <ResendVerification email={email} startCoolingDown /> : null}
      <p className="mt-6 text-center text-xs text-ink-subtle">
        Already confirmed?{" "}
        <Link to="/login" className="font-medium text-accent hover:underline">
          Sign in
        </Link>
      </p>
    </div>
  );
}
