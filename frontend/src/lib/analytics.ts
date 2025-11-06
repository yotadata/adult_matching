export type GAEventParams = Record<string, string | number | boolean | undefined>;

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
  }
}

const sanitizeParams = (params?: GAEventParams) => {
  if (!params) return undefined;
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null);
  return entries.length ? Object.fromEntries(entries) : undefined;
};

export const trackEvent = (name: string, params?: GAEventParams) => {
  if (typeof window === 'undefined') return;
  if (typeof window.gtag !== 'function') return;
  if (!name) return;
  const payload = sanitizeParams(params) ?? {};
  window.gtag('event', name, payload);
};

export const generateSessionId = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  const rand = Math.random().toString(16).slice(2, 10);
  return `sess_${Date.now()}_${rand}`;
};

export {};
