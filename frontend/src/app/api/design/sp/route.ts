import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function GET() {
  try {
    // Resolve the path to the repo-level docs/sp_design.svg
    const svgPath = path.resolve(process.cwd(), '..', 'docs', 'sp_design.svg');
    const data = await fs.readFile(svgPath, 'utf-8');
    return new NextResponse(data, {
      status: 200,
      headers: {
        'Content-Type': 'image/svg+xml; charset=utf-8',
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch {
    return NextResponse.json({ error: 'SVG not found' }, { status: 404 });
  }
}
