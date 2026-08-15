import { cn } from "@/lib/utils";

/** A label with a native hover tooltip. Data-forward: the explanation never takes up space. */
export function Tip({
  tip,
  children,
  className,
}: {
  tip: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      title={tip}
      className={cn(
        "cursor-help underline decoration-dotted decoration-muted-foreground/70 underline-offset-[3px]",
        className,
      )}
    >
      {children}
    </span>
  );
}
