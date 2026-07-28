import type { Metadata } from "next";
import TraceViewer from "./TraceViewer";

export const metadata: Metadata = {
  title: "PADOC Trace Viewer — Focus + Context Prototype",
  description:
    "An interactive prototype for exploring large distributed traces one call tree at a time.",
};

export default function Home() {
  return <TraceViewer />;
}
