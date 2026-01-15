 # Baseline (no perturbation)
  uv run inspect eval inspect_evals/gdm_cover_your_tracks \
    --model google/gemini-2.5-flash

  # With XML tags
  uv run inspect eval inspect_evals/gdm_cover_your_tracks \
    --model google/gemini-2.5-flash \
    -T prompt_perturbation=xml_tags

  # With deliberation prompt
  uv run inspect eval inspect_evals/gdm_cover_your_tracks \
    --model google/gemini-2.5-flash \
    -T prompt_perturbation=deliberate

  # Both combined
  uv run inspect eval inspect_evals/gdm_cover_your_tracks \
    --model google/gemini-2.5-flash \
    -T prompt_perturbation=xml_and_deliberate