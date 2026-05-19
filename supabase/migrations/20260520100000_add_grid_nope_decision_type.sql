alter table public.user_video_decisions
  drop constraint if exists user_video_decisions_decision_type_check;

alter table public.user_video_decisions
  add constraint user_video_decisions_decision_type_check
  check (decision_type in ('swipe_like', 'swipe_nope', 'grid_like', 'grid_nope'));
