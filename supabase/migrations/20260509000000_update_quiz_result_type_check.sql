alter table public.quiz_diagnosis_results
  drop constraint quiz_diagnosis_results_result_type_check;

delete from public.quiz_diagnosis_results
  where result_type not in (
    'spnc', 'spnw', 'spxc', 'spxw',
    'senc', 'senw', 'sexc', 'sexw',
    'mpnc', 'mpnw', 'mpxc', 'mpxw',
    'menc', 'menw', 'mexc', 'mexw'
  );

alter table public.quiz_diagnosis_results
  add constraint quiz_diagnosis_results_result_type_check
  check (result_type in (
    'spnc', 'spnw', 'spxc', 'spxw',
    'senc', 'senw', 'sexc', 'sexw',
    'mpnc', 'mpnw', 'mpxc', 'mpxw',
    'menc', 'menw', 'mexc', 'mexw'
  ));
