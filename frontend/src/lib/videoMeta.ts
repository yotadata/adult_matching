export const isUpcomingRelease = (value?: string | null): boolean => {
  if (!value) return false;
  const releaseDate = new Date(value);
  if (Number.isNaN(releaseDate.getTime())) return false;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const releaseDay = new Date(releaseDate);
  releaseDay.setHours(0, 0, 0, 0);
  return releaseDay.getTime() >= today.getTime();
};

export {};
