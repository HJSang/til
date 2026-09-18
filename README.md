# Today I Learned

This private repository is a durable record of what I learn from daily work,
reading, and other repositories.

## Daily routine

1. At the end of the day, open the GitHub issue created by the scheduled
   workflow (`Daily learning — YYYY-MM-DD`).
2. Handwrite one or more ideas first. Use the issue prompts only as a guide;
   the thinking and wording should be yours.
3. Transcribe the note into `daily/YYYY-MM-DD.md` and run the
   [grammar-only prompt](GRAMMAR_ONLY.md). Keep the corrected version close to
   your original voice and meaning.
4. Check the morning and before-sleep reread boxes when you actually reread a
   recent note.
5. On Saturday morning, use the `Weekly learning review — YYYY-MM-DD` issue to
   reread the last seven days and record connections, gaps, and next questions.

The reminders run in `America/Los_Angeles` time as closely as GitHub Actions
cron permits: daily at 9:00 PM during daylight time and weekly Saturday at
9:00 AM during daylight time. GitHub cron is UTC, so the displayed local time
shifts by one hour during standard time. Change the two cron expressions in
`.github/workflows/learning-reminders.yml` if you prefer a different hour.

## Note format

Start from [`daily/TEMPLATE.md`](daily/TEMPLATE.md). A good note is short:
one idea, why it matters, a concrete example, and one question or next step.
Topics can come from work, something you read, or another repository.

## Existing notes

**Total TILs: 2**

### Machine Learning

- [How Distributed Optimizer Checkpoints Work in Megatron](machine-learning/megatron-distributed-optimizer-checkpoints.md)
- [How Megatron-LM `save_checkpoint` Works](machine-learning/megatron-save-checkpoint.md)
