"use client";

/* Light / dark, kept in one place.
 *
 * The console is drawn dark, and that is right for a dim room and a 3-D well.
 * It is wrong for a projector, a photocopier and a printed page — inverted,
 * the type does not come out and the page drinks ink. So the theme is a real
 * setting: it lands on <html data-theme>, every colour in globals.css is a
 * token that reads from there, and the choice persists per browser.
 *
 * The initial value is applied by an inline script in layout.tsx, before the
 * first paint. This component only owns the switching, so a reload never
 * flashes the wrong palette.
 */

import { useEffect, useState } from "react";
import { Button } from "./ui";

export type Theme = "dark" | "light";

export const THEME_KEY = "inob-theme";

/** Read the theme the pre-paint script already resolved. */
function currentTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* Private mode: the theme still applies, it just will not persist. */
  }
}

/** Subscribe to the theme currently on <html>. */
export function useTheme(): Theme {
  const [theme, setTheme] = useState<Theme>("dark");
  useEffect(() => {
    const read = () => setTheme(currentTheme());
    read();
    // The attribute is written by the pre-paint script, by this component and
    // by the command palette, so watch the element rather than trying to route
    // every writer through one setter.
    const obs = new MutationObserver(read);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => obs.disconnect();
  }, []);
  return theme;
}

export default function ThemeToggle({ size = "md" }: { size?: "sm" | "md" }) {
  // The DOM is the source of truth: reading it during render would mismatch
  // hydration, so the hook corrects after mount.
  const theme = useTheme();
  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <Button
      icon="theme"
      size={size}
      onClick={() => applyTheme(next)}
      title={
        next === "light"
          ? "Switch to the light theme — the one that prints"
          : "Switch back to the dark console"
      }
      ariaLabel={`Switch to the ${next} theme`}
    >
      {theme === "dark" ? "Light" : "Dark"}
    </Button>
  );
}
