#!/bin/bash

d1=`date +"%Y-%m-%d %H:%M:%S"`

python simple.py $1 $2

d2=`date +"%Y-%m-%d %H:%M:%S"`

python3 notify_email.py go_jerk.sh "$d1" "$d2" ""

