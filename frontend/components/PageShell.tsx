import type { ReactNode } from "react";
import { BoomerangVideo } from "@/components/BoomerangVideo";
import { ButtonLink, cx } from "@/components/ui/primitives";

export function PageShell({
  title,
  description,
  eyebrow,
  actions,
  children,
  className,
  header,
}: {
  title: string;
  description: string;
  eyebrow: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  header?: ReactNode;
}) {
  return (
    <div className="relative isolate min-h-screen overflow-hidden px-4 py-4 text-foreground sm:px-6">
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <BoomerangVideo mode="blurred" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(4,6,9,0.44),rgba(4,6,9,0.58)_34%,rgba(4,6,9,0.74)_62%,rgba(4,6,9,0.88))]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(240,208,150,0.1),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(113,154,212,0.1),transparent_26%),radial-gradient(circle_at_right,rgba(255,255,255,0.04),transparent_20%)]" />
      </div>
      <div className="relative z-10 mx-auto flex max-w-7xl flex-col gap-8">
        <div className="flex justify-end">
          <ButtonLink href="/" variant="ghost">
            Homepage
          </ButtonLink>
        </div>
        {header}
        <main
          className={cx(
            "grid gap-8 rounded-[36px] border border-white/10 bg-[rgba(4,6,9,0.38)] px-4 py-6 backdrop-blur-[2px] sm:px-6 sm:py-8 pb-10",
            className,
          )}
        >
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div className="max-w-3xl space-y-3">
              <p className="font-mono text-xs uppercase tracking-[0.34em] text-[var(--accent-soft)]">
                {eyebrow}
              </p>
              <h1 className="font-display text-5xl leading-none text-[var(--foreground)] sm:text-6xl">
                {title}
              </h1>
              <p className="text-sm leading-7 text-[var(--muted)] sm:text-base">{description}</p>
            </div>
            {actions}
          </div>
          {children}
        </main>
      </div>
    </div>
  );
}
