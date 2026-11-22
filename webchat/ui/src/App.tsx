// The page: pick a run and checkpoint (header), then Chat (with the Setup drawer), Complete or Inspect. The three
// views stay mounted while hidden, so switching tabs keeps their state.
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { ChatView } from "./components/ChatView";
import { CompleteView } from "./components/CompleteView";
import { Header, type Tab } from "./components/Header";
import { InspectView } from "./components/inspect/InspectView";
import { SetupDrawer } from "./components/SetupDrawer";
import { applyToolSet, buildHeader, initialSetup, type Setup } from "./setup";
import type { Presets, Run, WorkspaceState } from "./types";

const DEFAULT_TOOLS = "shell + files + code";

export function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [current, setCurrent] = useState<Run | null>(null);
  const [ckpt, setCkpt] = useState("latest");
  const [presets, setPresets] = useState<Presets | null>(null);
  const [tab, setTab] = useState<Tab>("chat");
  const [setupOpen, setSetupOpen] = useState(window.innerWidth >= 900);
  const [setup, setSetup] = useState<Setup>(initialSetup);
  const [workspace, setWorkspace] = useState<WorkspaceState | null>(null);
  const [rawMode, setRawMode] = useState(false);
  const [draft, setDraft] = useState("");
  const [chatEpoch, setChatEpoch] = useState(0);  // a new chat: the chat view starts over
  const chatBusy = useRef(false);
  const chatDoc = useRef("");
  const forcedComplete = useRef(false);  // a completion run opened Complete: a chat run returns to Chat

  const examples = useMemo(() => {
    const cats: Record<string, string[]> = { ...(presets?.examples || {}) };
    for (const [k, v] of Object.entries(presets?.data_examples || {})) cats["From the data: " + k] = v;
    return cats;
  }, [presets]);

  const refreshWorkspace = () => { api.workspace().then(setWorkspace); };
  const newChat = () => { if (chatBusy.current) return; chatDoc.current = ""; setChatEpoch((e) => e + 1); };

  async function selectRun(name: string, list: Run[] = runs, wanted?: string | null) {
    const r = list.find((x) => x.run === name);
    if (!r) return;
    const changed = current?.run !== r.run;
    setCurrent(r);
    localStorage.setItem("run", r.run);
    setCkpt(wanted && r.checkpoints.includes(wanted) ? wanted : r.checkpoints[0]);
    if (!changed) return;
    const p = await api.presets(r.dataset);
    setPresets(p);
    const firstCat = Object.keys(p.examples)[0] ?? "";
    setSetup((s) => ({ ...s, toolSet: DEFAULT_TOOLS, checked: applyToolSet(s, p, DEFAULT_TOOLS), exCat: firstCat }));
    newChat();
    const completion = r.mode === "completion";
    setTab((t) => {
      if ((completion && t === "chat") || (!r.inspect && t === "inspect")) { forcedComplete.current = true; return "complete"; }
      if (!completion && forcedComplete.current && t === "complete") { forcedComplete.current = false; return "chat"; }
      if (!completion) forcedComplete.current = false;
      return t;
    });
  }

  useEffect(() => {
    api.runs().then((list) => {
      setRuns(list);
      const saved = localStorage.getItem("run");
      const pick = list.find((r) => r.run === saved) || list.find((r) => r.arch.startsWith("min-gru")) || list[0];
      if (pick) selectRun(pick.run, list, localStorage.getItem("ckpt:" + pick.run));
    });
    refreshWorkspace();
  }, []);

  return (
    <>
      <Header runs={runs} current={current} ckpt={ckpt} tab={tab} setupOpen={setupOpen}
              onSelectRun={(name) => selectRun(name)}
              onCkpt={(c) => { if (current) localStorage.setItem("ckpt:" + current.run, c); setCkpt(c); newChat(); }}
              onTab={(t) => { setTab(t); if (t === "complete") forcedComplete.current = false; }}
              onNewChat={() => { newChat(); setTab("chat"); }}
              onToggleSetup={() => setSetupOpen(!setupOpen)} />
      <div className="body">
        <ChatView key={chatEpoch} visible={tab === "chat"} current={current} ckpt={ckpt} presets={presets} setup={setup}
                  examples={examples} draft={draft} onDraft={setDraft} rawMode={rawMode} docRef={chatDoc}
                  onBusy={(b) => { chatBusy.current = b; }}
                  onTemp={(temp) => setSetup((s) => ({ ...s, gen: { ...s.gen, temp } }))}
                  onWorkspaceChanged={refreshWorkspace} />
        <CompleteView visible={tab === "complete"} current={current} ckpt={ckpt} />
        <InspectView visible={tab === "inspect"} current={current} ckpt={ckpt}
                     chatDoc={() => chatDoc.current || buildHeader(setup, presets)} />
        {tab === "chat" && setupOpen && (
          <SetupDrawer presets={presets} setup={setup} onSetup={setSetup} examples={examples}
                       onExample={(t) => { setDraft(t); setTab("chat"); }}
                       workspace={workspace} onWorkspaceChanged={refreshWorkspace}
                       rawMode={rawMode} onRawMode={(r) => { setRawMode(r); setTab("chat"); }} />
        )}
      </div>
    </>
  );
}
