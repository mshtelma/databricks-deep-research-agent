export interface HostFieldProps {
  value: string;
  onChange: (v: string) => void;
  label?: string;
  hint?: string;
}

export function HostField({
  value,
  onChange,
  label = 'Target workspace host',
  hint,
}: HostFieldProps) {
  return (
    <div>
      <label
        style={{
          display: 'block',
          fontSize: 12,
          fontWeight: 500,
          color: 'var(--db-navy-800)',
          marginBottom: 5,
        }}
      >
        {label}
      </label>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="https://acme-prod.cloud.databricks.com"
        style={{
          width: '100%',
          border: '1px solid var(--db-gray-lines)',
          background: '#fff',
          borderRadius: 6,
          padding: '7px 10px',
          fontFamily: 'var(--font-mono-db)',
          fontSize: 12,
          color: 'var(--db-navy-800)',
          outline: 'none',
          transition: 'border-color 120ms ease-out, box-shadow 120ms ease-out',
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = 'var(--db-navy-400)';
          e.currentTarget.style.boxShadow = 'var(--db-shadow-focus)';
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = 'var(--db-gray-lines)';
          e.currentTarget.style.boxShadow = 'none';
        }}
      />
      {hint && (
        <div style={{ fontSize: 11, color: 'var(--db-gray-text)', marginTop: 4 }}>
          {hint}
        </div>
      )}
    </div>
  );
}
