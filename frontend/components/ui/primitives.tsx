import Link from "next/link";
import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  Ref,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from "react";
import { forwardRef } from "react";

export function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

type ButtonVariant = "primary" | "secondary" | "ghost";

const buttonStyles: Record<ButtonVariant, string> = {
  primary:
    "border border-[rgba(255,245,220,0.28)] bg-[linear-gradient(180deg,rgba(241,215,172,0.96),rgba(230,184,108,0.82))] text-black shadow-[0_20px_60px_rgba(230,184,108,0.24)] backdrop-blur-xl hover:bg-[linear-gradient(180deg,rgba(247,227,193,0.98),rgba(235,194,124,0.9))]",
  secondary:
    "border border-white/16 bg-[linear-gradient(180deg,rgba(255,255,255,0.16),rgba(255,255,255,0.06))] text-[var(--foreground)] shadow-[inset_0_1px_0_rgba(255,255,255,0.14)] backdrop-blur-2xl hover:bg-[linear-gradient(180deg,rgba(255,255,255,0.2),rgba(255,255,255,0.08))]",
  ghost:
    "border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.09),rgba(255,255,255,0.03))] text-[var(--muted-strong)] shadow-[inset_0_1px_0_rgba(255,255,255,0.1)] backdrop-blur-2xl hover:bg-[linear-gradient(180deg,rgba(255,255,255,0.14),rgba(255,255,255,0.05))]",
};

type BaseActionProps = {
  children: ReactNode;
  variant?: ButtonVariant;
  className?: string;
};

export function ButtonLink({
  children,
  href,
  variant = "primary",
  className,
}: BaseActionProps & { href: string }) {
  return (
    <Link
      href={href}
      className={cx(
        "inline-flex min-h-11 items-center justify-center rounded-full px-5 text-sm font-semibold tracking-[0.18em] uppercase transition duration-300",
        buttonStyles[variant],
        className,
      )}
    >
      {children}
    </Link>
  );
}

export function Button({
  children,
  variant = "primary",
  className,
  ...props
}: BaseActionProps & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={cx(
        "inline-flex min-h-11 items-center justify-center rounded-full px-5 text-sm font-semibold tracking-[0.18em] uppercase transition duration-300",
        buttonStyles[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export const SurfaceCard = forwardRef(function SurfaceCard(
  {
    children,
    className,
  }: {
    children: ReactNode;
    className?: string;
  },
  ref: Ref<HTMLDivElement>,
) {
  return (
    <div ref={ref} className={cx("panel-glow mac-glass rounded-[28px] p-5", className)}>
      {children}
    </div>
  );
});

export function SectionHeading({
  eyebrow,
  title,
  description,
  align = "left",
}: {
  eyebrow: string;
  title: string;
  description?: string;
  align?: "left" | "center";
}) {
  return (
    <div className={cx("space-y-3", align === "center" && "text-center")}>
      <p className="font-mono text-xs uppercase tracking-[0.34em] text-[var(--accent-soft)]">
        {eyebrow}
      </p>
      <h2 className="font-display text-4xl leading-none text-[var(--foreground)] sm:text-5xl">
        {title}
      </h2>
      {description ? (
        <p className="max-w-2xl text-sm leading-7 text-[var(--muted)] sm:text-base">
          {description}
        </p>
      ) : null}
    </div>
  );
}

export function Pill({
  children,
  active = false,
  onClick,
}: {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <span
      onClick={onClick}
      className={cx(
        "inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-medium tracking-[0.12em] uppercase backdrop-blur-xl",
        onClick && "cursor-pointer",
        active
          ? "border-[rgba(255,245,220,0.24)] bg-[linear-gradient(180deg,rgba(241,215,172,0.92),rgba(230,184,108,0.78))] text-black shadow-[0_10px_30px_rgba(230,184,108,0.18)]"
          : "border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.05))] text-[var(--muted-strong)] shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]",
      )}
    >
      {children}
    </span>
  );
}

export function StatTile({
  label,
  value,
  trend,
}: {
  label: string;
  value: string;
  trend?: string;
}) {
  return (
    <SurfaceCard className="h-full">
      <div className="space-y-2">
        <p className="text-sm text-[var(--muted)]">{label}</p>
        <p className="font-display text-4xl leading-none text-[var(--foreground)]">{value}</p>
        {trend ? <p className="text-sm text-[var(--accent-soft)]">{trend}</p> : null}
      </div>
    </SurfaceCard>
  );
}

export function StateBlock({
  title,
  description,
  tone = "neutral",
}: {
  title: string;
  description: string;
  tone?: "neutral" | "success" | "warning" | "danger";
}) {
  const toneStyles = {
    neutral:
      "border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.04))] text-[var(--muted-strong)]",
    success:
      "border-[rgba(122,215,168,0.26)] bg-[linear-gradient(180deg,rgba(122,215,168,0.16),rgba(122,215,168,0.05))] text-[var(--success)]",
    warning:
      "border-[rgba(241,198,107,0.26)] bg-[linear-gradient(180deg,rgba(241,198,107,0.16),rgba(241,198,107,0.05))] text-[var(--warning)]",
    danger:
      "border-[rgba(255,141,141,0.26)] bg-[linear-gradient(180deg,rgba(255,141,141,0.16),rgba(255,141,141,0.05))] text-[var(--danger)]",
  } as const;

  return (
    <div
      className={cx(
        "panel-glow rounded-[24px] border p-4 shadow-[0_12px_40px_rgba(0,0,0,0.14)] backdrop-blur-2xl",
        toneStyles[tone],
      )}
    >
      <p className="text-sm font-semibold">{title}</p>
      <p className="mt-1 text-sm leading-6 text-[var(--muted)]">{description}</p>
    </div>
  );
}

function fieldBaseClass(className?: string) {
  return cx(
    "themed-field mac-glass w-full rounded-[20px] px-4 py-3 text-sm text-[var(--foreground)] outline-none transition placeholder:text-white/30 focus:border-[var(--border-strong)] focus:bg-white/[0.18]",
    className,
  );
}

export function TextInput({
  label,
  helper,
  error,
  className,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & {
  label: string;
  helper?: string;
  error?: string;
}) {
  return (
    <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
      <span>{label}</span>
      <input className={fieldBaseClass(className)} {...props} />
      {helper ? <span className="text-xs text-[var(--muted)]">{helper}</span> : null}
      {error ? <span className="text-xs text-[var(--danger)]">{error}</span> : null}
    </label>
  );
}

export function SelectField({
  label,
  children,
  className,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement> & {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
      <span>{label}</span>
      <select className={fieldBaseClass(cx("themed-select", className))} {...props}>
        {children}
      </select>
    </label>
  );
}

export function TextAreaField({
  label,
  helper,
  className,
  ...props
}: TextareaHTMLAttributes<HTMLTextAreaElement> & {
  label: string;
  helper?: string;
}) {
  return (
    <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
      <span>{label}</span>
      <textarea className={fieldBaseClass(className)} {...props} />
      {helper ? <span className="text-xs text-[var(--muted)]">{helper}</span> : null}
    </label>
  );
}

export function CheckboxField({
  label,
  description,
  checked,
  defaultChecked,
  onChange,
}: {
  label: string;
  description: string;
  checked?: boolean;
  defaultChecked?: boolean;
  onChange?: (checked: boolean) => void;
}) {
  return (
    <label className="mac-glass flex items-start gap-3 rounded-[24px] p-4 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        defaultChecked={defaultChecked}
        onChange={onChange ? (e) => onChange(e.target.checked) : undefined}
        className="mt-1 h-4 w-4 rounded border-white/20 bg-transparent accent-[var(--accent)]"
      />
      <span className="space-y-1">
        <span className="block text-sm font-medium text-[var(--foreground)]">{label}</span>
        <span className="block text-sm leading-6 text-[var(--muted)]">{description}</span>
      </span>
    </label>
  );
}

export function RangeSlider({
  label,
  value,
  defaultValue = 55,
  onChange,
}: {
  label: string;
  value?: number;
  defaultValue?: number;
  onChange?: (value: number) => void;
}) {
  return (
    <label className="mac-glass grid gap-3 rounded-[24px] p-4">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="text-[var(--muted-strong)]">{label}</span>
        <span className="font-mono text-xs uppercase tracking-[0.2em] text-[var(--accent-soft)]">
          {value ?? defaultValue}%
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={value}
        defaultValue={value === undefined ? defaultValue : undefined}
        onChange={onChange ? (e) => onChange(Number(e.target.value)) : undefined}
        className="accent-[var(--accent)]"
      />
    </label>
  );
}

export function SkeletonCard() {
  return (
    <SurfaceCard className="space-y-4">
      <div className="h-4 w-20 animate-pulse rounded-full bg-white/[0.16] backdrop-blur-xl" />
      <div className="h-8 w-3/4 animate-pulse rounded-full bg-white/[0.16] backdrop-blur-xl" />
      <div className="h-4 w-full animate-pulse rounded-full bg-white/[0.14] backdrop-blur-xl" />
      <div className="h-4 w-5/6 animate-pulse rounded-full bg-white/[0.14] backdrop-blur-xl" />
    </SurfaceCard>
  );
}

export function ListPanel({
  title,
  items,
}: {
  title: string;
  items: string[];
}) {
  return (
    <SurfaceCard className="h-full">
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-[var(--foreground)]">{title}</h3>
        <ul className="space-y-3 text-sm text-[var(--muted)]">
          {items.map((item) => (
            <li
              key={item}
              className="rounded-[18px] border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(255,255,255,0.04))] px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-xl"
            >
              {item}
            </li>
          ))}
        </ul>
      </div>
    </SurfaceCard>
  );
}
