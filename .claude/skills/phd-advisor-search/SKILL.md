---
name: phd-advisor-search
description: Research and rank potential PhD advisors in Israel (or elsewhere) for a given dissertation topic. Use when the user wants a shortlist of academic advisors with verified current activity, institution, recent publications, research focus, and former-student outcomes. Triggers on requests like "find me PhD advisors for X", "who should I approach for a doctorate in Y", "סקירת מנחים לדוקטורט".
---

# PhD Advisor Search

Builds a verified, ranked shortlist of potential PhD advisors for a given research topic, using the `deep-research` skill's coordinator/researcher/report-writer pattern. Produces real, sourced profiles — never invented names, papers, or affiliations.

## When to use

The user wants to identify potential doctoral advisors/supervisors for a specific research topic and wants, for each candidate: current activity status, institutional affiliation, recent publications, a research-focus summary, and information about former doctoral/postdoc students.

## Required inputs (ask the user if not given)

Before starting, confirm:
1. **Research topic** — the dissertation's subject matter.
2. **Disciplinary orientation** — which field(s) should be the PRIMARY lens, and which methods/tools (if any) are SECONDARY (e.g., "public health / trauma / women's health primary, machine learning secondary — not the core of the dissertation"). This matters a lot: a secondary-tool framing should bias candidate search toward substantive-field researchers who use that tool, not toward tool specialists who lack the substantive focus.
3. **Geographic / institutional scope** — country and which institutions count (e.g., "only the 7 major Israeli research universities: Hebrew University of Jerusalem, Tel Aviv University, Technion, Weizmann Institute of Science, Ben-Gurion University of the Negev, Bar-Ilan University, University of Haifa" — vs. all institutions, vs. a specific list).
4. **Definition of "active"** — e.g., a publication within the last N years (2-3 is a reasonable default), and whether to separately flag anyone with an EXPLICIT statement that they are currently recruiting PhD students (recommended: yes, since almost no one states this and it's a strong bonus signal when found).
5. **List size** — how many ranked candidates the user wants (default: 20).
6. **LinkedIn usage for former-student info** — there is no LinkedIn connector/API access. The only option is best-effort web search that may surface public, search-engine-indexed LinkedIn snippets (name, title, institution). This is partial and must always be labeled "(LinkedIn snippet, unverified)" when used — never presented as confirmed. Confirm the user is OK with this limitation.
7. **Output format** — a one-time report, a reusable skill (this file, already reusable for future reruns with updated criteria), or both.

## Process

1. **Decompose the topic into independent candidate-discovery angles.** Don't search generically — split the topic into 3-5 mutually-exclusive thematic/methodological clusters that together cover it. For a topic like "trauma in reservist women/families using ML as a secondary tool," good clusters were: (a) clinical trauma/PTSD researchers, (b) researchers specifically on military/reservist FAMILIES, (c) public health & women's health researchers, (d) researchers combining the substantive field with the secondary tool (ML). Adapt this decomposition to whatever topic is given.

2. **Round 1 — candidate discovery.** Spawn one parallel research subagent per cluster (see `deep-research` skill's researcher pattern). Each subagent should return 8-15 REAL candidates with: name, confirmed current institution/department/rank (via the institution's own page where possible), at least one recent (within-window) publication as evidence of activity, a short focus description, and any explicit accepting/not-accepting-students signal. Explicitly instruct subagents: never fabricate a name, paper, or affiliation; if something can't be verified, say so rather than guess; exclude anyone who cannot be confirmed as currently active unless clearly flagged as uncertain.

3. **Consolidate and deduplicate** the candidate pool across all Round-1 reports (the same person often surfaces from multiple angles — that overlap is itself a useful relevance signal). Group the deduplicated list into batches of ~5-6 people each, by institution or theme, for deep verification.

4. **Round 2 — deep verification (the deep-research skill's one-additional-round allowance).** Spawn one parallel subagent per batch. For EACH person, require:
   - Confirmed current status (confirmed current / likely current / could not confirm), with institution and exact department
   - 2-3 sentence research-focus summary tailored to the user's stated orientation (what overlaps with the primary field, what overlaps with the secondary tool)
   - The 5 most recent peer-reviewed publications (title, year, venue), in reverse chronological order — if fewer than 5 can be verified, say so explicitly rather than padding
   - Any explicit "currently recruiting PhD students" statement, quoted/cited if found
   - Up to 5 former PhD/postdoc students and what they do now, via Google Scholar co-authorship patterns, dissertation databases, lab alumni pages, and best-effort web search (labeling anything LinkedIn-snippet-sourced as unverified)
   Instruct subagents explicitly to downgrade or recommend excluding anyone whose verified current work turns out to have no real connection to the topic, even if they looked promising in Round 1 — honesty over padding the list.

5. **Report writer.** Spawn a single report-writer subagent (per `deep-research`'s pattern) with the full required structure: intro + methodology caveats, a quick-scan summary table, full ranked profiles (all required fields, tiered/ranked by fit), and an explicit "considered but not included" appendix (retired/emeritus people, people with no real topical connection despite seniority, people just outside the institutional scope who are nonetheless highly relevant — flag these as exceptions worth knowing about). Write the report in the user's language.

6. **Deliver.** Save the report, and if the user wants it as a lasting project artifact (not just a chat/session report), copy it into the relevant project folder and commit it.

## Known limitations to always disclose to the user

- Web-page fetching (direct site reads) may be blocked in the research environment; candidate verification then relies on search-engine snippets only, which can leave some publication lists incomplete and some details (exact rank/department) conflicting across sources. Always flag this rather than presenting incomplete data as exhaustive.
- There is no LinkedIn login/scraping capability. Former-student current-role information is therefore best-effort and should be verified by the user directly before relying on it.
- No faculty member's page reliably states whether they are accepting PhD students — treat any explicit statement as a bonus signal, and recommend the user email candidates directly to confirm availability.
- Academic rank, department names, and affiliation status (active vs. emeritus) sometimes conflict across sources (personal CV/Academia.edu pages lag behind official university directories) — the official university page should be treated as authoritative when in conflict.

## Re-running this skill

To re-run with updated criteria (e.g., a new year, a different list size, additional institutions, or a narrower sub-topic), invoke this skill again and answer the "Required inputs" questions with the updated parameters — no changes to this file are needed for a routine rerun.
