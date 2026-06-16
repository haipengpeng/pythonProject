#!/bin/bash

#LOGFILE="2025-05-08.2.debug.log"
LOGFILE="2025-05-12.19.debug.log" 

function calc_cost {
  local key="$1"
  local max=$(grep "$key" "$LOGFILE" | awk -F"$key: |ms" '{print $2}' | sort -nr | head -1)
  if [[ -z "$max" ]]; then max=0; fi
  local avg=$(grep "$key" "$LOGFILE" | awk -F"$key: |ms" '{print $2}' | \
    awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
  echo "$key max: $max ms"
  echo "$key avg: $avg ms"
  echo
}

#calc_cost "nlpc text server cost"
#calc_cost "word matcher cost"
#calc_cost "check text, cost"

calc_cost "TEST::: finish to check preset, userName: xingnengmodi, cost"
calc_cost "TEST::: finish to calc premerge,  cost"
calc_cost "TEST::: finish to calc preset,  cost"

calc_cost "TEST::: finish to filter str,  cost"
calc_cost "TEST::: finish to limit str,  cost"

calc_cost "TEST::: finish to search db,  cost"

calc_cost "TEST::: finish to pre text,  cost"
calc_cost "TEST::: finish to real text,  cost"
calc_cost "TEST::: finish to generate text,  cost"
calc_cost "TEST::: finish to get text res,  cost"

calc_cost "TEST::: finish to pre wordMatcher,  cost"
calc_cost "TEST::: finish to real wordMatcher,  cost"
calc_cost "TEST::: finish to generate wordMatcher,  cost"
calc_cost "TEST::: finish to get res wordMatcher,  cost"

calc_cost "TEST::: finish to merge,  cost"

calc_cost "TEST::: finish to all suanzi,  cost"
calc_cost "TEST::: finish to save obj,  cost"
calc_cost "TEST::: finish to save es,  cost"

