# Structured observation V3

V3 keeps the 1,500-value legacy vector for compatibility and adds categorical,
count-aware inputs. New structured checkpoints are not shape-compatible with V2.

## Information boundary

- `own_deck_ids` is the acting player's submitted 60-card list.
- No opponent deck list or hidden opponent hand/prize identity is exposed.
- `prize_ids` contains only identities actually returned by the game API; facedown
  cards remain zero.
- `aux_target` may use simulator truth as a supervised training target, but it is
  never fed into the actor's state encoder.

## New card zones

- `prize_ids[2, 6]`: known prize identities from each relative perspective.
- `search_ids[60]`: cards offered by the current deck-search selection.
- `looking_ids[60]`: cards currently revealed by a look effect.
- `own_deck_ids[60]`: initial own deck composition.
- `context_card_ids[3]`: context card, resolving effect card, stadium.

Hand, discard, prize, search, looking, own-deck and log sets use mean, maximum,
scaled sum and explicit count pooling. The sum/count terms preserve multiplicity.

## Static card relations

Each card representation includes up to three printed attacks, up to three
individual skill-effect vectors, its own-name evolution token and its
`evolvesFrom` token. Attack costs, attack text effects and energy roles remain
separate inputs.

## Option features

`option_features` has 21 columns:

0. relative player
1. source index
2. in-play target index
3. selected number
4. selected count
5. already selected in an autoregressive selection
6. resolved card identity exists
7. attack identity exists
8. printed attack damage
9. base damage adjusted for printed weakness/resistance
10. opposing Active remaining HP
11. immediate base KO indicator
12. prizes from that base KO
13. target is Pokémon ex
14. total legal option count
15. options of the same type
16. options resolving to the same card identity
17. current hand occupancy
18. current deck occupancy
19. free Bench capacity
20. option is an Ability

Conditional damage, effects from other cards and random outcomes are not claimed
to be exact by columns 9-12. The policy must combine their static effect features
with state or later receive a dedicated rules evaluator.
