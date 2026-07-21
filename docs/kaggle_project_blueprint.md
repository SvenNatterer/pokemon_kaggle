# Kaggle-Projekt-Blaupause: der derzeit belegte Erfolgs-Stack

Stand: 2026-07-21. Diese Seite ist eine übertragbare Startvorlage für ein neues
Kaggle-RL-/Agentenprojekt. Sie trennt bewusst **belegte Ergebnisse** von
plausiblen, aber noch nicht auf Kaggle bestätigten Verbesserungen.

## 0. Ehrliche Ausgangslage

**War das Modell auf Kaggle schon stark?**  
Es war konkurrenzfähig, aber noch nicht konstant stark: die beste bestätigte
Live-Submission war *V6 Compact Alakazam 1M* mit **572,4 End-Elo und 50,9 %
Siegen in 55 Spielen**. Die späteren vier gemessenen Submissions endeten bei
484--542 Elo. Kaggle ist damit der externe Realitätscheck, nicht der Beweis,
dass eine starke lokale Validierung automatisch gewinnt.

**Was ist lokal der beste belegte Kandidat?**  
`ppo_v6_deck_bank_70_compact_c` gewann die aktuelle Auswahl: **89,0 %** auf
210 Validierungsspielen (Wilson-Untergrenze 84,1 %, keine technischen Fehler)
und **84,4 %** auf 180 separaten Holdout-Spielen (Wilson 78,4 %, keine
technischen Fehler). Sein schlechtestes Holdout-Matchup lag trotzdem nur bei
46,7 %. Daher ist Robustheit gegen Lücken im Metagame der nächste Hebel.

## 1. Ein-Satz-Architektur

**Was bauen?**  
Einen kleinen, rekurrenten PPO-Agenten mit maskierten legalen Aktionen,
strukturierten Objekt-/Kartenmerkmalen und einem separaten Modell der
verborgenen Information; trainiert gegen eine diverse, adaptive Gegnerliga und
ausgewählt ausschließlich über eine eingefrorene Validierungsliga.

```text
Spiel-Engine -> legaler Aktionsraum + strukturierte Beobachtung -> PPO + LSTM
      ^                                                       |
      |                         Gegnerliga <- PFSP-lite <----+
      |
Arena/Replays (Diagnose) <- Validation (Auswahl) <- Holdout (einmaliger Bericht)
                                      |
                                  Kaggle-Upload
```

## 2. Die wichtigen Bausteine -- kurz beantwortet

| Frage | Bewährte Antwort | Warum es zählt |
| --- | --- | --- |
| Was ist die wichtigste Modellentscheidung? | Nur **legal auswählbare** Aktionen anbieten und sie vor dem Sampling maskieren. | Der Agent verschwendet keine Lernkapazität auf unmögliche Züge; technische Fehler werden messbar. |
| Wie groß soll der Aktionsraum sein? | Kompakt: 65 auswählbare Optionen plus eine Stop-Aktion (= 66). | Die aktuelle V6-Compact-Linie ersetzt den früheren 1.000er Raum. |
| Welche Eingaben? | Spielstand, Objekte auf dem Feld, Hand/Discard/Log, Optionen samt Ziel und Karte, plus Karten-/Attack-Metadaten. IDs werden als Kategorien eingebettet, nicht als Zahlen interpretiert. | Die Entscheidung sieht sowohl Zustand als auch Bedeutung jeder Option. |
| Wie geht das Modell mit unbekannten Karten um? | Hilfskopf sagt verdeckte Kartenanzahlen voraus; 64-dimensionale Belief-Einbettung geht in den Actor. | Ein praktisches Gedächtnis für unvollständige Information, ohne verborgene Ziele bei der Inferenz zu leaken. |
| Warum LSTM? | Spielentscheidungen hängen von Zugfolge, bereits gezeigten Karten und Mehrfachauswahlen ab. | Der Actor kann Verlauf statt nur Momentaufnahme nutzen. |
| Was ist das kleinste sinnvolle Netz? | `compact`: ca. 4,82 Mio. Policy-Parameter, ca. 4,04k kombinierte Features. | Das ist wesentlich kleiner als `full` (13,3 Mio. Parameter / 20k Features) und der aktuelle Standard. |
| Wie wird belohnt? | Gewinn +1, Niederlage -1; zusätzlich potentialbasiert kleiner Fortschritt aus Preis-Karten und HP-Differenz. | Dichter Lernimpuls, ohne das Endziel zu ersetzen. Bei \(gamma=0,999\) bleibt der Shaping-Term policy-invariant. |
| Gegen wen trainieren? | Gegen eine **Mischung** aus PPO-Snapshots, historischen Modellen und regelbasierten Gegnern, nie nur gegen einen Spiegelgegner. | Verhindert Overfitting auf eine Spielweise und macht Kaggle-Überraschungen weniger fatal. |
| Wie wird die Gegnerverteilung angepasst? | PFSP-lite alle 200 Episoden: fokussiere unsichere, ungefähr ausgeglichene Matchups; behalte 20 % Zufall und max. 35 % pro Gegner. | Lernt an lehrreichen Schwächen, ohne die Vielfalt zu verlieren. |
| Wie vermeidet man Startspieler-Bias? | Perspektive pro Episode zufällig drehen und in der Evaluation exakt ausgleichen. | Eine hohe Winrate wird nicht mit Spieler-0-Vorteil verwechselt. |
| Welche Fehler sind disqualifizierend? | Jeder Engine-Fehler, ungültige Learner-Aktion oder Aktionsraum-Overflow. | Ein Modell mit guter Score, aber technischen Aussetzern, wird nicht befördert. |
| Wie wird ausgewählt? | Zuerst Health-Gate, dann Wilson-95%-Untergrenze, schlechtestes Matchup, Gesamtscore. | Priorisiert robuste Untergrenze statt Zufallstreffer. |
| Wozu dienen Replays? | Nur Diagnose: Verlustgrund, Gegnertyp und konkrete Fehlentscheidung suchen. | Arena-Elo und Replays erzeugen Hypothesen; sie entscheiden nicht allein über Promotion. |
| Was gehört in die Submission? | Modell, Deck/Artefakt, Inferenzcode, benötigte Runtime und Linux-x86_64-Engine; vor Upload isoliert testen. | Paket-/Runtime-Fehler können bessere Policies vollständig zunichtemachen. |

## 3. Empfohlenes Startprofil (heutiger Standard)

Dies ist der reproduzierbare V6-Compact-Ausgangspunkt, kein Dogma:

```yaml
algorithm: recurrent PPO
policy: V6 compact, structured observations, categorical card/attack embeddings
actions: 66 (65 legal option slots + stop)
memory: LSTM + belief actor (belief_dim: 64, belief gradients detached)
auxiliary_loss: hidden-card counts, coefficient: 0.10
reward: terminal ±1 + potential shaping
gamma: 0.999
parallel_envs: 7
rollout_per_env: 2048
batch_size: 1024
ppo_epochs: 2
learning_rate: 0.0001
entropy_coefficient: 0.008
clip_range: 0.12
target_kl: 0.03
base_budget: 1_000_000 steps
fine_tune_budget: 400_000 steps
perspective_rotation: true
card_table_cache: true
health_gate: strict / zero tolerance
```

**Wann nicht blind übernehmen?**  
Wenn die Umgebung kaum verborgene Information, keine Mehrschritt-Aktionen oder
einen kleinen festen Aktionsraum besitzt, sind Belief-Head, LSTM oder die
komplexe Optionenkodierung möglicherweise unnötig. Dann zuerst eine Ablation
gegen dieses Startprofil laufen lassen.

## 4. Projektstruktur zum Kopieren

```text
new-kaggle-agent/
  src/
    env.py                 # Engine-Adapter, legale Optionen, Rewards
    observation.py         # stabile Feature-Spezifikation + Encoder
    policy.py              # Embeddings, Extractor, LSTM, Hilfsköpfe
    train.py               # einziger Trainingseinstieg
    evaluate.py            # eingefrorene Liga, Wilson, Perspektiven
    health.py              # Fehlerzähler und Promotion-Gate
    arena/                 # kontinuierliche Diagnoseliga und Replays
  configs/
    training_pool.json
    validation_manifest.json
    final_holdout_manifest.json
    factory.json
  models/
    foundation/ validation/ holdout/ stage_snapshots/ experiments/
  experiments/YYYY-MM/
    <run>.sh               # vollständiger, wiederholbarer Aufruf
  reports/
  artifacts/submissions/
  scripts/
    build_submission.sh
    verify_submission_package.py
  tests/
    test_action_space.py test_observation.py test_inference.py
```

**Unverhandelbare Regel:** Trainings-, Validierungs- und Holdout-Gegner sind
disjunkte, versionierte Manifeste. Die Validierung darf Modellwahl beeinflussen;
der finale Holdout wird erst nach der Wahl einmal gelesen. Kaggle-Replays werden
als eigene externe Evidenz gespeichert, niemals still in den Trainingspool
gemischt.

## 5. Konkreter Ablauf für ein neues Projekt

1. **Kontrakt zuerst:** Definiere Observation, Aktion, Terminierung und
   Inferenz-API. Schreibe Paritäts-Tests zwischen Training und Submission.
2. **Sichere Baseline:** Baue einen legalen, deterministischen Regelbot und
   einen kleinen PPO ohne Extras. Er muss das Paket und alle Smoke-Tests bestehen.
3. **Feature- und Aktionsraum:** Ersetze rohe numerische IDs durch Embeddings,
   baue die Aktionsmaske ein und protokolliere jede ungültige Aktion.
4. **Diverse Liga:** Erzeuge mindestens 6--10 stilistisch unterschiedliche
   Gegner (Regelbots, ältere PPOs, verschiedene Decks/Archetypen). Reserviere
   zusätzlich je mindestens 6 valide und finale Gegner.
5. **Mehrere Foundations:** Trainiere mindestens drei unterschiedliche Seeds/
   Startzustände; wähle sie auf der Validierung nach Untergrenze, nicht nach
   Bestwert.
6. **Gezielt feintrainieren:** Pro Hypothese nur eine Änderung: etwa
   Trainingspool, Feature, Reward oder Budget. Erzeuge unveränderliche
   Stage-Snapshots samt Git-Revision, Seed, Manifest-Hashes und Endschritt.
7. **Auswahl:** 100 Spiele pro Gegner normal, 200 bei engem Ergebnis, beide
   Perspektiven gleich oft. Verwerfe jede Variante, die das Health-Gate nicht
   besteht oder eine Perspektivlücke über 10 Prozentpunkte hat.
8. **Finaler Bericht:** Einmalig auf dem versiegelten Holdout. Danach bei
   weiterem Tuning einen *neuen* Holdout erstellen.
9. **Kaggle-Release:** Archiv bauen, auf Linux-x86_64 prüfen, isolierten Agenten
   testen, Upload mit Modell-/Deck-/Git-Hash dokumentieren. Danach Elo-Verlauf
   und Replays auswerten.

## 6. Was die Ergebnisse tatsächlich stützen

**Gut gestützt**

- Der Wechsel auf V6 mit kompaktem, legal maskiertem 66er Aktionsraum und
  strukturierten Merkmalen ist der derzeitige Betriebsstandard.
- Eine kleine Compact-Policy ist praktisch und leistungsfähig: der aktuelle
  lokale Sieger war auf Validation 89,0 % / Wilson 84,1 % und auf Holdout
  84,4 % / Wilson 78,4 %, jeweils ohne technische Fehler.
- Diverse eingefrorene Ligen, Perspektiv-Ausgleich, Wilson-Untergrenze und ein
  hartes Health-Gate machen Auswahlentscheidungen aussagekräftiger als Arena-Elo
  oder einzelne Siege.
- PFSP-lite war lokal leicht besser als statisches Sampling: 85,3 % gegenüber
  84,3 % auf derselben 700-Spiel-Validierung. Beide Varianten hatten jedoch
  Overflows und wären deshalb nach dem heutigen Gate nicht promotierbar.
- Regelbasierte Gegner sind echte starke Baselines, nicht nur Fallbacks:
  Abomasnow lag lokal bei 86,2 % auf 210 Validierungsspielen.

**Noch nicht als Kaggle-Gewinn behaupten**

- Lokale Spitzenwerte haben noch keine stabile Steigerung der Live-Elo
  nachgewiesen. Der beste Abschluss lag bei 572,4 Elo; die nachfolgenden
  Compact/PFSP-Submissions lagen bei 542, 512 und 484 Elo.
- Mehr Trainingszeit allein half nicht zuverlässig: die PFSP-Endless-Variante
  fiel lokal gegen ein neues schwieriges Matchup auf 66,7 % (schlechtester
  Gegner 21 %). Vielfalt und frische Gegner sind wichtiger als bloßes Weiterlernen.
- Der aktuelle finale Holdout enthält Alakazam auch in der Validation. In einem
  neuen Projekt muss diese Überschneidung ausgeschlossen werden -- sonst ist
  der "finale" Bericht teilweise nicht mehr unabhängig.

## 7. Diagnose-Playbook nach einem Kaggle-Run

| Befund | Wahrscheinlichste Reaktion |
| --- | --- |
| Viele ungültige Züge/Overflows | Aktionsenkodierung und Maskierung reparieren; nicht weitertrainieren oder promoten. |
| Hohe lokale Score, schwache Kaggle-Elo | Trainingsliga um beobachtete Gegnertypen erweitern, aber Kaggle-Daten als separaten, versionierten externen Pool behandeln. |
| Einzelnes katastrophales Matchup | Replay lesen, eine Ursache formulieren und genau eine Pool-/Feature-/Reward-Änderung ablatieren. |
| Große Spieler-0/Spieler-1-Differenz | Startperspektiven ausgleichen, Observation/Perspektivspiegelung testen; Score nicht als Fortschritt werten. |
| Häufiges Deck-out | Kartenziehen, Deckgröße und lange Partien explizit als Zustand/Reward- bzw. Gegnerligenthema untersuchen. |
| Submission startet nicht oder wird langsam | Paket isoliert starten, Threads auf 1 setzen, Linux-Binary prüfen und schnellen legalen Fallback vorsehen. |

## 8. Minimaler Run-Record pro Modell

```yaml
model_id: <name>
git_revision: <commit>
seed: <integer>
base_model: <hash-or-none>
feature_schema: <version + hash>
action_schema: <version + hash>
training_manifest: <sha256>
validation_manifest: <sha256>
holdout_manifest: <sha256>
command: <exact command>
timesteps: <completed steps>
health:
  engine_errors: 0
  invalid_actions: 0
  option_overflows: 0
validation:
  games_per_opponent: 100
  wilson_lb: <value>
  worst_matchup: <value>
  perspective_gap: <value>
submission_archive_sha256: <hash>
kaggle_submission_id: <id>
```

## 9. Startentscheidung

Wenn heute ein neues Projekt begonnen wird, starte mit dem Profil aus Abschnitt
3 und investiere die ersten Iterationen in (1) perfekte Train-/Inference-Parität,
(2) eine diverse und wirklich disjunkte Gegnerliga und (3) automatische,
fehlertolerante Submission-Checks. Erst danach lohnt sich Architektur- oder
Reward-Forschung. Das sind die Elemente, die den jetzigen Stand reproduzierbar
machen -- und die Lücke zwischen lokaler Stärke und Kaggle-Ergebnis am ehesten
schließen.
