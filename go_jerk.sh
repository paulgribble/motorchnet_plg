#!/bin/bash

d1=`date +"%Y-%m-%d %H:%M:%S"`

python simple.py

d2=`date +"%Y-%m-%d %H:%M:%S"`

python3 notify_email.py go_jerk.sh "$d1" "$d2" ""

