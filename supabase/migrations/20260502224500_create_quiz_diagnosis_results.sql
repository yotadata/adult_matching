create table public.quiz_diagnosis_results (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  result_type text not null,
  gender text not null,
  axis_scores jsonb not null,
  answers jsonb not null,
  created_at timestamptz not null default now(),
  constraint quiz_diagnosis_results_result_type_check check (result_type in (
    'snph', 'snpl', 'sneh', 'snel',
    'sxph', 'sxpl', 'sxeh', 'sxel',
    'mnph', 'mnpl', 'mneh', 'mnel',
    'mxph', 'mxpl', 'mxeh', 'mxel'
  )),
  constraint quiz_diagnosis_results_gender_check check (gender in ('male', 'female', 'other'))
);

create index if not exists idx_quiz_diagnosis_results_created_at
  on public.quiz_diagnosis_results (created_at desc);

create index if not exists idx_quiz_diagnosis_results_result_type_created_at
  on public.quiz_diagnosis_results (result_type, created_at desc);

create index if not exists idx_quiz_diagnosis_results_user_id_created_at
  on public.quiz_diagnosis_results (user_id, created_at desc);

alter table public.quiz_diagnosis_results enable row level security;

create policy "insert quiz diagnosis results"
on public.quiz_diagnosis_results for insert
to anon, authenticated
with check (
  user_id is null or auth.uid() = user_id
);

create policy "select own quiz diagnosis results"
on public.quiz_diagnosis_results for select
to authenticated
using (
  auth.uid() = user_id
);
