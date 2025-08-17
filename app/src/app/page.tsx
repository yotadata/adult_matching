import Header from "@/components/Header";
import SwipeArea from "@/components/SwipeArea";

export default function Home() {
  return (
    <main className="relative w-screen h-screen overflow-hidden pt-20">
      <Header />
      <SwipeArea />
    </main>
  );
}
