#!/bin/bash

d1=`date +"%Y-%m-%d %H:%M:%S"`
SECONDS=0

#python model.py 1 paul_test

duration=$SECONDS
d2=`date +"%Y-%m-%d %H:%M:%S"`

python notify_email.py motornet "$d1" "$d2" "$duration" /Users/plg/Documents/Data/paul_test/learning_curve.png
