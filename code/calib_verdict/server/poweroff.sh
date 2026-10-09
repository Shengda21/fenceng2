#!/bin/bash
date -u +%FT%TZ > /root/jobs/poweroff_time.txt
sync
/usr/bin/shutdown
