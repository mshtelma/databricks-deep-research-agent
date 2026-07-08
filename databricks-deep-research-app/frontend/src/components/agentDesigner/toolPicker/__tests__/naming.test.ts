import { describe, expect, it } from 'vitest';

import { sanitizeToolName, suggestedToolName, uniqueToolName } from '../naming';

describe('sanitizeToolName', () => {
  it('lowercases and snake_cases arbitrary input', () => {
    expect(sanitizeToolName('Pct Change!')).toBe('pct_change');
    expect(sanitizeToolName('__weird---name__')).toBe('weird_name');
  });

  it('never returns an empty or digit-leading name', () => {
    expect(sanitizeToolName('###')).toBe('tool');
    expect(sanitizeToolName('42_things')).toBe('fn_42_things');
  });
});

describe('suggestedToolName', () => {
  it('uses the FQN tail for UC functions', () => {
    expect(suggestedToolName('uc_function', 'main.metrics.pct_change')).toBe('pct_change');
  });

  it('uses the attribute for module:attr imports', () => {
    expect(suggestedToolName('decorated', 'my_pkg.tools:normalize_text')).toBe(
      'normalize_text',
    );
  });

  it('falls back to the kind when there is no target', () => {
    expect(suggestedToolName('web_search')).toBe('web_search');
  });
});

describe('uniqueToolName', () => {
  const existing = [{ name: 'pct_change' }, { name: 'pct_change_2' }];

  it('returns the base when free', () => {
    expect(uniqueToolName('other', existing)).toBe('other');
  });

  it('suffixes past every taken name', () => {
    expect(uniqueToolName('pct_change', existing)).toBe('pct_change_3');
  });
});
