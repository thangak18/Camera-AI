import type { ReactNode } from "react";

type Props = {
  icon: ReactNode;
  label: string;
  value: number | string;
};

export function Metric({ icon, label, value }: Props) {
  return (
    <div className="metric">
      <div className="metric-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

