alter table public.quiz_diagnosis_results
  add column if not exists anonymous_session_id text;

update public.quiz_diagnosis_results
set anonymous_session_id = coalesce(anonymous_session_id, gen_random_uuid()::text)
where anonymous_session_id is null;

alter table public.quiz_diagnosis_results
  alter column anonymous_session_id set not null;

create index if not exists idx_quiz_diagnosis_results_anonymous_session_id_created_at
  on public.quiz_diagnosis_results (anonymous_session_id, created_at desc);
