import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-white">
      <h2 className="text-4xl font-bold mb-4">404 Not Found</h2>
      <p className="text-lg mb-8">Could not find requested resource</p>
      <Link href="/" className="px-6 py-3 bg-pink-500 rounded-full text-white font-bold">
        Return Home
      </Link>
    </div>
  );
}
