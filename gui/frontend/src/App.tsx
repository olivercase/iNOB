import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alignment,
  Button,
  Intent,
  Navbar,
  Tag,
} from "@blueprintjs/core";
import { ConfigDict, getHealth, HealthResponse, saveConfig } from "./api";
import { AppToaster } from "./toaster";
import ConfigEditor from "./components/ConfigEditor";
import Viewer3D from "./components/Viewer3D";
import RunConsole from "./components/RunConsole";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [consoleOpen, setConsoleOpen] = useState(true);
  const [saving, setSaving] = useState(false);
  const configRef = useRef<ConfigDict | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealthError(true));
  }, []);

  const onConfigChange = useCallback((cfg: ConfigDict) => {
    configRef.current = cfg;
  }, []);

  async function handleSave() {
    if (!configRef.current) return;
    setSaving(true);
    try {
      const r = await saveConfig(configRef.current);
      const t = await AppToaster;
      if (r.saved) t.show({ intent: Intent.SUCCESS, message: "Config saved." });
      else
        t.show({
          intent: Intent.DANGER,
          message: `Save rejected: ${r.errors.join("; ")}`,
        });
    } catch (e) {
      (await AppToaster).show({ intent: Intent.DANGER, message: String(e) });
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <Navbar>
        <Navbar.Group align={Alignment.LEFT}>
          <Navbar.Heading>Vagus-FM</Navbar.Heading>
          <Navbar.Divider />
          <Button
            minimal
            icon="play"
            intent={Intent.SUCCESS}
            onClick={() => setConsoleOpen(true)}
          >
            Run
          </Button>
          <Button
            minimal
            icon="floppy-disk"
            loading={saving}
            onClick={handleSave}
          >
            Save Config
          </Button>
        </Navbar.Group>
        <Navbar.Group align={Alignment.RIGHT}>
          {healthError ? (
            <Tag intent={Intent.DANGER} minimal icon="error">
              backend offline
            </Tag>
          ) : health ? (
            <Tag intent={Intent.SUCCESS} minimal icon="pulse">
              {health.status} · {health.active_config ?? "default"}
            </Tag>
          ) : (
            <Tag minimal icon="time">
              connecting…
            </Tag>
          )}
        </Navbar.Group>
      </Navbar>

      <div className="app-body">
        <div className="app-left">
          <Viewer3D />
        </div>
        <div className="app-right">
          <ConfigEditor onConfigChange={onConfigChange} />
        </div>
      </div>

      <RunConsole open={consoleOpen} onToggle={() => setConsoleOpen((o) => !o)} />
    </>
  );
}
