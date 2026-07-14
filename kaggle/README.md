# Kaggle V6 training

1. Build the dataset bundle from the current local source and selected bases:

   ```bash
   venv/bin/python scripts/build_kaggle_v6_bundle.py --owner-slug YOUR_KAGGLE_USERNAME
   kaggle datasets create -p kaggle_bundle
   ```

2. Import `pokemon_v6_training.ipynb` into Kaggle and attach the created
   `pokemon-kaggle-v6-training-bundle` dataset.

3. Add a private Kaggle secret named `WANDB_API_KEY`. If it is absent, the
   notebook deliberately falls back to offline W&B logging.

4. Adjust only the configuration cell. The default fine-tunes `bank_63` from
   Base B for 400,000 steps. This is local factory target 15 of 16, so Kaggle
   should finish well before the local queue reaches it. Set `RUN_KIND = "base"`
   for a fresh one-million-step
   V6 base.

5. Save a notebook version after training. The final model, experiment record,
   logs and W&B offline data are collected under `/kaggle/working/artifacts`.

6. Copy the resulting `ppo_v6_deck_bank_63_base_b.zip` and its sibling
   `ppo_v6_deck_bank_63_base_b.complete` into `models/opponent_factory_v6/`
   before the local queue reaches target 15. The local factory will then reuse
   the completed Kaggle result instead of training the same target again.
