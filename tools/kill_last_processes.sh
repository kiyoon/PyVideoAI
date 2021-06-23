#!/bin/bash

proc_ids="$(cat last_pids.txt)"

while read proc_id
do
	kill $proc_id
done <<< "${proc_ids}"
