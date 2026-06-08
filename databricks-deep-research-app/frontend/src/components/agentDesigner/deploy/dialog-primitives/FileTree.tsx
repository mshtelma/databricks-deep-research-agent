export interface FileTreeEntry {
  path: string;
  size?: string;
  note?: string;
}

export interface FileTreeProps {
  files: FileTreeEntry[];
}

export function FileTree({ files }: FileTreeProps) {
  return (
    <pre
      style={{
        margin: 0,
        padding: '10px 14px',
        fontFamily: 'var(--font-mono-db)',
        fontSize: 12,
        color: 'var(--db-navy-800)',
        background: 'var(--db-oat-light)',
        border: '1px solid var(--db-gray-lines)',
        borderRadius: 6,
        overflow: 'auto',
        lineHeight: 1.6,
      }}
    >
      {'📁 root/\n'}
      {files.map((f, i) => {
        const connector = i < files.length - 1 ? '├── ' : '└── ';
        const sizeNote = [f.size, f.note].filter(Boolean).join('  ');
        return connector + f.path + (sizeNote ? '  ' + sizeNote : '') + '\n';
      })}
    </pre>
  );
}
