/// <reference types="vite/client" />
import { describe, it, expect } from 'vitest';

declare const __dirname: string;

describe('Responsive Layout (style.css)', async () => {
  const nodePrefix = 'node:';
  const fs: any = await import(`${nodePrefix}fs`);
  const path: any = await import(`${nodePrefix}path`);
  const cssPath = path.resolve(__dirname, '../style.css');
  const cssContent: string = fs.readFileSync(cssPath, 'utf-8');

  it('clamps figures-grid minmax track size to container width', () => {
    expect(cssContent).toMatch(/\.figures-grid\s*\{[^}]*grid-template-columns:\s*repeat\(\s*auto-fit,\s*minmax\(\s*min\(100%,\s*280px\),\s*1fr\)\s*\)/);
  });

  it('does not contain hardcoded minmax 320px in figures-grid', () => {
    expect(cssContent).not.toMatch(/\.figures-grid\s*\{[^}]*minmax\(\s*320px/);
  });

  it('contains 480px mobile media query for figures-grid single column layout', () => {
    const mobileQueryRegex = /@media\s*\(\s*max-width:\s*480px\s*\)\s*\{[\s\S]*?\.figures-grid\s*\{[^}]*grid-template-columns:\s*1fr;[\s\S]*?\}/;
    expect(cssContent).toMatch(mobileQueryRegex);
  });
});
