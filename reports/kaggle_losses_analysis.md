# Kaggle Replay Loss Analysis

This report analyzes the downloaded Kaggle replays, focusing specifically on our advanced submissions (Elo > 450).

## Submission Performance Overview

| Submission ID | Description | Total Games | Win Rate | Wins | Losses | Draws |
| --- | --- | --- | --- | --- | --- | --- |
| 54405445 | Deck 1 | 42 | 47.6% | 20 | 22 | 0 |
| 54423382 | Deck 100 Agent | 48 | 50.0% | 24 | 24 | 0 |
| 54430612 | Deck 0 Agent | 44 | 36.4% | 16 | 28 | 0 |
| 54495680 | v4 Base Bain Fixed Dimension Mismatch and Policy Override bugs | 28 | 46.4% | 13 | 15 | 0 |
| 54495882 | Deck 9 Old V3 Model (Dim 100) | 14 | 21.4% | 3 | 11 | 0 |
| 54498832 | v4 abra deck 9 | 21 | 28.6% | 6 | 15 | 0 |
| 54499398 | Deck 7 (Tournament Deck) | 57 | 42.1% | 24 | 33 | 0 |
| 54505397 | Deck Bank 47 (Abra) | 55 | 40.0% | 22 | 33 | 0 |
| 54567482 | Deck 18 | 30 | 33.3% | 10 | 19 | 1 |
| 54582753 | V5 Lucario | 73 | 38.4% | 28 | 45 | 0 |
| 54617196 | V5b Mega Lucario ex 3 | 84 | 42.9% | 36 | 48 | 0 |
| 54723890 | Base A | V6 energy scaling + Linux x86_64 | validation 87.9% | 70 | 38.6% | 27 | 43 | 0 |
| 54784364 | V6 Compact Alakazam 1M 2026-07-17 | 55 | 49.1% | 27 | 28 | 0 |
| 54793285 | V6 Compact bank_54 scratch PFSP endless 2026-07-17 | 45 | 57.8% | 26 | 19 | 0 |

## High-Tier Submissions (>450 Elo) Loss Analysis
Analyzed submissions: 54723890, 54784364, 54793285

### Loss Reasons Distribution

| Loss Reason | Count | Percentage |
| --- | --- | --- |
| Prize KO | 70 | 77.8% |
| Deck Out | 20 | 22.2% |

### Losses by Opponent Archetype

| Opponent Archetype | Count | Percentage |
| --- | --- | --- |
| Alakazam | 17 | 18.9% |
| Dragapult ex | 10 | 11.1% |
| Mega Lucario ex | 10 | 11.1% |
| Mega Abomasnow ex | 9 | 10.0% |
| Mega Kangaskhan ex | 8 | 8.9% |
| Marnie's Grimmsnarl ex | 7 | 7.8% |
| Crustle | 5 | 5.6% |
| Dudunsparce | 4 | 4.4% |
| Team Rocket's Mewtwo ex | 4 | 4.4% |
| Mega Starmie ex | 3 | 3.3% |
| Archaludon ex | 2 | 2.2% |
| Cynthia's Garchomp ex | 2 | 2.2% |
| Raging Bolt ex | 2 | 2.2% |
| Terapagos ex | 1 | 1.1% |
| Hop's Trevenant | 1 | 1.1% |
| Mega Lopunny ex | 1 | 1.1% |
| Ethan's Typhlosion | 1 | 1.1% |
| Iono’s Bellibolt ex | 1 | 1.1% |
| N’s Zoroark ex | 1 | 1.1% |
| Mega Gengar ex | 1 | 1.1% |

### Average Duration (Turns) by Loss Reason

| Loss Reason | Avg Turns | Range |
| --- | --- | --- |
| Prize KO | 123.8 | 18-237 |
| Deck Out | 156.0 | 103-243 |

### Detailed Matchups & Loss Reasons

| Opponent Archetype | Loss Reason | Count |
| --- | --- | --- |
| Alakazam | Prize KO | 15 |
| Dragapult ex | Prize KO | 10 |
| Mega Lucario ex | Prize KO | 9 |
| Mega Abomasnow ex | Prize KO | 8 |
| Marnie's Grimmsnarl ex | Prize KO | 7 |
| Mega Kangaskhan ex | Deck Out | 6 |
| Crustle | Prize KO | 4 |
| Mega Starmie ex | Prize KO | 3 |
| Team Rocket's Mewtwo ex | Prize KO | 3 |
| Mega Kangaskhan ex | Prize KO | 2 |
| Dudunsparce | Deck Out | 2 |
| Raging Bolt ex | Prize KO | 2 |
| Alakazam | Deck Out | 2 |
| Dudunsparce | Prize KO | 2 |
| Mega Lucario ex | Deck Out | 1 |
| Archaludon ex | Prize KO | 1 |
| Crustle | Deck Out | 1 |
| Cynthia's Garchomp ex | Prize KO | 1 |
| Terapagos ex | Deck Out | 1 |
| Hop's Trevenant | Deck Out | 1 |
| Mega Lopunny ex | Prize KO | 1 |
| Team Rocket's Mewtwo ex | Deck Out | 1 |
| Mega Abomasnow ex | Deck Out | 1 |
| Archaludon ex | Deck Out | 1 |
| Ethan's Typhlosion | Deck Out | 1 |
| Cynthia's Garchomp ex | Deck Out | 1 |
| Iono’s Bellibolt ex | Prize KO | 1 |
| N’s Zoroark ex | Deck Out | 1 |
| Mega Gengar ex | Prize KO | 1 |
