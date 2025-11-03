'use client';

import Image from 'next/image';
import React from 'react';

export default function MobileDesignPreview() {
  return (
    <div className="min-h-screen w-full flex flex-col items-center bg-gray-50">
      <div className="w-full max-w-6xl px-4 py-6">
        <h1 className="text-xl font-semibold mb-4">Mobile Design Preview</h1>
        <p className="text-sm text-gray-600 mb-4">
          This page renders <code>docs/sp_design.svg</code> via an internal API for quick review.
        </p>
        <div className="w-full overflow-auto rounded-lg border border-gray-200 bg-white shadow-sm">
          <Image
            src="/api/design/sp"
            alt="Mobile Design SVG"
            width={1200}
            height={2400}
            className="w-full h-auto block"
            priority
            unoptimized
          />
        </div>
      </div>
    </div>
  );
}
