"use client";

// The keyboard contract, written down. Every binding listed here is bound in
// page.tsx; if one moves, this list moves with it.

import { Sheet } from "@/components/ui";

const GROUPS: { name: string; rows: [string, string][] }[] = [
  {
    name: "Anywhere",
    rows: [
      ["⌘K  /  Ctrl K", "Open the command palette"],
      ["?", "Show these shortcuts"],
      ["Esc", "Close whatever is open, one layer at a time"],
    ],
  },
  {
    name: "The run",
    rows: [
      ["R", "Run the journey, or stop one in flight"],
      ["L", "Show or hide the run log"],
      ["C", "Compare forward models"],
      [",", "Advanced settings"],
    ],
  },
  {
    name: "The canvas",
    rows: [
      ["A", "Add an output figure"],
      ["F", "Fit the journey to the view"],
      ["Click", "Open a card’s step"],
      ["Drag", "Move a card; drag the background to pan"],
      ["Scroll", "Zoom about the pointer"],
      ["Arrows", "Nudge the selected card (hold Shift for a bigger step)"],
      ["Delete", "Remove the selected output card"],
    ],
  },
];

export default function ShortcutsHelp({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  return (
    <Sheet open={open} onClose={onClose} title="Keyboard shortcuts" icon="terminal">
      <div className="drawer-body">
        <p className="drawer-lead ui-dim">
          Shortcuts are ignored while you are typing in a field, so a stray
          letter never starts a run.
        </p>
        {GROUPS.map((g) => (
          <section key={g.name} className="jkeys">
            <h3>{g.name}</h3>
            <dl>
              {g.rows.map(([key, what]) => (
                <div key={key} className="jkeys-row">
                  <dt>
                    <kbd className="jkbd">{key}</kbd>
                  </dt>
                  <dd>{what}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </Sheet>
  );
}
