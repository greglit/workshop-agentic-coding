WORKSHOP
# Agentic Coding
## Grundlagen, Open-Source-Lösungen und lokale Modelle

Gregor Wolf

---

## Repo-Übersicht

Dieses Repo enthält alle Materialien des Workshops:
- **slides/** &mdash; Präsentation als PDF und HTML-File
- **live-demo/** &mdash; Zwei kleine Python-Skripte die Token-By-Token zeigen wie ein LLM zum Agenten wird
- **opencode.json.example** &mdash; Beispielkonfiguration für OpenCode mit Ollama und Atlassian MCP

---

# Workshop Aufgaben
Hier sind einige Dinge zusammengestellt die alleine oder in Teams ausprobiert werden können. Alle Aufgaben sind freiwillig und nicht zwingend von anderen abhängig.

Jeder Abschnitt kann auch mit Hilfe eines Agents bearbeitet werden. Dazu einfach Agent-Prompt kopieren und in ClaudeCode oder ähnlich einfügen.

## Inhaltsverzeichnis

- [OpenCode](#opencode)
- [OpenChamber](#openchamber)
- [OpenWork](#openwork)
- [Lokale Modelle](#lokale-modelle)
- [Atlassian MCP](#atlassian-mcp)
- [Skills](#skills)
- [Weiterführende Links](#weiterführende-links)

## OpenCode und Co.
Hier kannst du zwischen 3 Optionen wählen. Wenn du den Terminal magst, installiere **OpenCode** direkt. Wenn du lieber eine grafische Benutzeroberfläche magst, dann spring direkt zu **OpenChamber**. Wenn du bisher Claude Cowork benutzt hast, kannst du **OpenWork** mal ausprobieren.

### OpenCode

#### Installation

Agent-Prompt in ClaudeCode kopieren
```
Richte OpenCode auf meinem Rechner ein. Folge den Setup-Schritten in https://github.com/greglit/workshop-agentic-coding/blob/main/README.md unter OpenCode. Gebe mir danach anhand der README einen Überblick über nächste Schritte.
```
**oder** mit Homebrew installieren:
```bash
brew install anomalyco/tap/opencode
```
**oder** mit curl:
```bash
curl -fsSL https://opencode.ai/install | bash
```

#### Erste Schritte
- In Terminal in einen Projektordner navigieren und mit `opencode` starten (Nicht den Benutzerordner und keinen mit sensiblen Daten!)
- Mit `/models` ein kostenloses Modell von OpenCode Zen wählen zB.: `Nemotron 3 Ultra Free`
- Mit `/connect` andere Provider anbinden (funktioniert nicht für lokale Ollama-Modelle)
- mit `Tab-Taste` in den Plan-Modus wechseln
- In `~/.config/opencode/` die `opencode.json.example` aus diesem Repo einfügen, `.example` aus dem Dateinamen entfernen und die Berechtigungen überprüfen
- Erster Prompt: z.B. `Gebe mir einen Überblick über den aktuellen Ordner`

### OpenChamber

#### Installation

Agent-Prompt in ClaudeCode kopieren
```
Richte OpenChamber auf meinem Rechner ein. Folge den Setup-Schritten in https://github.com/greglit/workshop-agentic-coding/blob/main/README.md unter OpenChamber. Gebe mir danach anhand der README einen Überblick über nächste Schritte.
```
**oder** von [openchamber.dev/download](https://openchamber.dev/download/) herunterladen und in den Programme-Ordner ziehen.

**oder** mit Terminal via Homebrew:
```bash
brew install --cask openchamber
```

#### Erste Schritte
- Nach Appstart einen Projektordner auswählen (Nicht den Benutzerordner und keinen mit sensiblen Daten!)
- Im Eingabefeld neben `Build` auf `Big Pickle` klicken und ein anderes Modell wählen zB.: `Nemotron 3 Ultra Free`
- Im selben Menü `+ Add new Provider` klicken um andere Provider anzubinden
- mit `Tab-Taste` in den Plan-Modus wechseln
- Unten links aufs `Zahnrad` klicken und unter `Agents` die Berechtigungen überprüfen
- Erster Prompt: z.B. `Gebe mir einen Überblick über den aktuellen Ordner`

### OpenWork

#### Installation

Agent-Prompt in ClaudeCode kopieren
```
Install OpenWork on my computer, set up my first workspace, and open it ready to use. Follow the steps in https://openworklabs.com/start.md?v=hero When finished show me next steps from https://github.com/greglit/workshop-agentic-coding/blob/main/README.md
```

**oder** von [openworklabs.com/download](https://openworklabs.com/download) herunterladen und in den Programme-Ordner ziehen.

#### Erste Schritte
- Nach Appstart links bei `Workspaces`über `+` einen Projektorner auswählen (Nicht den Benutzerordner und keinen mit sensiblen Daten!)
- Im Eingabefeld auf `Big Pickle` klicken und ein anderes Modell wählen zB.: `Nemotron 3 Ultra Free`
- Im selben Menü `Connect more Providers` klicken um andere Provider anzubinden
- Überprüfen ob im aktuellen Ordner Dinge sind, die nicht kaputt gehen sollen!
- Wenn nein, dann erster Prompt: z.B. `Gebe mir einen Überblick über den aktuellen Ordner`

---
## Lokale Modelle

### Ollama installieren

Agent-Prompt in OpenCode/OpenChamber/OpenWork kopieren
```
Installiere Ollama und lade ein lokales Modell für Agentic Coding herunter. Folge der Anleitung "Lokale Modelle" in https://github.com/greglit/workshop-agentic-coding/blob/main/README.md. Kläre vorhandene Hardware ab und bespreche die Modellwahl mit dem User, bevor du ein Modell herunterlädst. Führe dann die Anbindung an OpenCode so wie in der README beschrieben durch.
```

**oder** von [ollama.com/download](https://ollama.com/download) herunterladen und in den Programme-Ordner ziehen.

**oder** im Terminal:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### LLM herunterladen

Empfehlungen für MacBooks (kleinere Modelle sind schneller, größere schlauer):

Für 8 GB RAM:
- `gemma3:1b-it-qat` (815 MB)

Für 16 GB RAM oder mehr:
- `gemma4:e2b` (7.2 GB)
- `gemma4:e4b` (9.6 GB)
- `gemma4:12b` (12 GB)
- `qwen3.5:9b` (6.6 GB)

Für 32 GB RAM oder mehr:
- `qwen3.8:27b` (18 GB)

Im Terminal
```bash
ollama run gemma4:e2b
```
danach `control`+`D` zum Beenden

Mehr Modelle auf [Ollama Library](https://ollama.com/library/)

### An OpenCode anbinden
1. Finder öffnen, `shift`+`command`+`G` drücken, `~/.config/opencode/` eingeben und `Enter`
2. `opencode.json.example` aus dem Repository runterladen, dort einfügen und `.example` aus dem Dateinamen entfernen
3. Datei im Editor öffnen und überprüfen ob die heruntergeladenen Modelle in der Liste sind
4. In OpenCode/OpenChamber/OpenWork Modell aus Liste auswählen


Alternativ zu Ollama: [LM Studio](https://lmstudio.ai/) GUI, Modellvorauswahl, umfangreiches Chat-Interface


## Atlassian MCP

**Setup via Agent**
```
Richte den Atlassian MCP für OpenCode ein. Folge den Setup-Schritten in https://github.com/greglit/workshop-agentic-coding/blob/main/README.md unter Atlassian MCP. Gebe mir danach anhand der README einen Überblick über nächste Schritte.
```

**oder manuell:**
1. Finder öffnen, `shift`+`command`+`G` drücken, `~/.config/opencode/` eingeben und `Enter`
2. `opencode.json.example` aus dem Repository runterladen, dort einfügen und `.example` aus dem Dateinamen entfernen
3. (Datei im Editor öffnen und API Key eingeben)
4. Überprüfen ob der MCP verfügbar ist:
  - OpenCode: `/mcps` eingeben
  - OpenChamber: Unten links aufs `Zahnrad` klicken und unter `MCP` nachsehen
  - OpenWork: Links auf `Library` klicken und zu `MCP-Server` runterscrollen

Beispiel-Prompt:
```
Erstelle mit Hilfe des Confluence-MCPs eine Confluence-Seite zum Inhalt des aktuellen Ordners.
```
Weitere Anregungen:
- Jira Ticket zu Confluence Seite Agent/Skill
- Confluence Seite zu Jira Tickets  Agent/Skill
- tldr.s Schreiben
- Seiten/Tickets überprüfen und auf Vollständigkeit testen.
- Automatische Recherche zu Ticket/Page
- Automatischer Plan zu Ticket/Page
- Meeting Notes im Confluence anlegen und Überprüfen.


## Skills

**Setup via Agent:**
```
Ich möchte mich mit Skills vertraut machen. Lies dir als erstes den Abschnitt Skills unter https://github.com/greglit/workshop-agentic-coding/blob/main/README.md durch. Gebe mir eine Übersicht über die dort beschriebenen Möglichkeiten und unterstütze mich bei der Einrichtung und Umsetzung.
```

Skills können von Plattformen wie [skills.sh](https://www.skills.sh/) github und co. bezogen werden oder selbst erstellt werden.

#### Skills von skills.sh installieren
Im Terminal
```
npx skills add https://github.com/vercel-labs/skills --skill find-skills
```
**oder** manuell Markdown-File von [hier](https://github.com/vercel-labs/skills/blob/main/skills/find-skills/SKILL.md) herunterladen und in `~/.config/opencode/skills/` unter neuem Ordner `find-skills` ablegen.

Danach beispielweise Prompten:
```
/find-skills Ich möchte einen Skill installieren der mir dabei hilft über das Atlassian-MCP Jira und Confluence zu bedienen.
```

#### OpenCode
- `/skills` zeigt installierte Skills
- keinen integrierten Mechanismus zum Skills finden und installieren

#### OpenChamber
- In den Einstellungen unter `Skills` installierte Skills einsehen und eigene Skills erstellen und bearbeiten
- In den Einstellungen unter `Skills Catalog` neue Skills aus festen Quellen beziehen und eigene Quellen hinzufügen

#### Eigenen Skill erstellen
1. Installiere einen "Skill-Creator"-Skill der dir dabei hilft neue Skills zu erstellen.

Prompt: 
```
/find-skills Ich möchte einen Skill installieren, der mir dabei hilft neue Skills zu erstellen.
```
oder in OpenChambers Skills Catalog nach `skill-creator` suchen.

2. Prompten mit neuem Skill "skill-creator":
```
/skill-creator Erstelle einen Skill indem wir definieren, was du tuhen sollst, wenn ich dich nach einer ausführlichen Recherche zu einem Thema frage.
```
```
/skill-creator Erstelle mir aus dem Arbeitsablauf dieser Session einen neuen Skill.
```



---

## Weiterführende Links

- [OpenCode Docs](https://opencode.ai/docs/)
- [OpenCode Guide](https://ai.sulat.com/the-definitive-guide-to-opencode-from-first-install-to-production-workflows-aae1e95855fb)
- [Hermes Agent (Nous Research)](https://github.com/nousresearch/hermes-agent)
- [Pi](https://github.com/earendil-works/pi) | [Oh My Pi](https://github.com/can1357/oh-my-pi)
- [Will It Run AI (Modell-Empfehlungen)](https://willitrunai.com/blog/best-ai-models-for-mac-16gb)
- [Local.AI Hardware & Benchmarks](https://local.ai/greglit/invite)
- [LLM Configurator GPU-Guide](https://llmconfigurator.com/en/guides/best-gpu-buyer-guide)
- [Artificial Analysis Leaderboard](https://artificialanalysis.ai/leaderboards/models)
- [LiveBench](https://livebench.ai/)
