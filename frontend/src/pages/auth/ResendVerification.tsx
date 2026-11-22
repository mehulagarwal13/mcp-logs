import { useEffect, useState } from "react";
import { resendVerification } from "@/api/auth";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/context/ToastContext";
import type { ApiError } from "@/types/common";

// Courtesy cooldown only. The backend enforces the real limits (per-row
// cooldown and a daily cap in the database, plus a per-IP limiter), so this
// timer cannot be relied on -- or bypassed -- for anything security related.
const RESEND_COOLDOWN_SECONDS = 60;

interface ResendVerificationProps {
  email: string;
  /** Start in cooldown (e.g. immediately after signup already sent one). */
  startCoolingDown?: boolean;
}

/**
 * "Resend the confirmation email" button. The server answers identically
 * whether or not a message was sent, so the copy never claims delivery or
 * says anything about whether the address is registered.
 */
export function ResendVerification({ email, startCoolingDown = false }: ResendVerificationProps) {
  const { toast } = useToast();
  const [secondsLeft, setSecondsLeft] = useState(startCoolingDown ? RESEND_COOLDOWN_SECONDS : 0);
  const [isSending, setIsSending] = useState(false);

  useEffect(() => {
    if (secondsLeft <= 0) return;
    const timer = window.setTimeout(() => setSecondsLeft((s) => s - 1), 1000);
    return () => window.clearTimeout(timer);
  }, [secondsLeft]);

  async function handleResend() {
    setIsSending(true);
    try {
      await resendVerification(email);
      toast({
        variant: "success",
        title: "Check your inbox",
        description: "If this address can be used, a new confirmation email is on its way.",
      });
      setSecondsLeft(RESEND_COOLDOWN_SECONDS);
    } catch (err) {
      const apiError = err as ApiError;
      const description =
        apiError?.status === 429
          ? "Please wait a moment before requesting another email."
          : apiError?.errorCode === "email.not_configured"
            ? "Email is not available right now. Please try again later."
            : "Couldn't send the email. Please try again shortly.";
      toast({ variant: "error", title: "Resend failed", description });
    } finally {
      setIsSending(false);
    }
  }

  return (
    <Button
      type="button"
      variant="secondary"
      className="w-full"
      isLoading={isSending}
      disabled={isSending || secondsLeft > 0 || !email}
      onClick={handleResend}
    >
      {secondsLeft > 0 ? `Resend email (${secondsLeft}s)` : "Resend confirmation email"}
    </Button>
  );
}
