# Content
This directory contains sample script to utilize GCP preemptible VM instances.

1. Copy following two files to home directory.
 * startup-script
 * shutdown-script

2. Set up meta data of GCP (Google Cloud Platform) as follows (for example):
 * key:   startup-script
 * value: following two lines
```sh
#!/bin/bash
sudo -u itsukara21 /home/itsukara21/startup-script >> /var/log/startup.log 
```

 * key:   shutdown-script
 * value: following two lines
```sh
#!/bin/bash
sudo -u itsukara21  /home/itsukara21/shutdown-script >> /var/log/shutdown.log
```

3. Set up Google Cloud SDK in non-preemptible VM (non-GCP VM is OK)
 * See [Google Cloud SDK Quickstarts for Linux](https://cloud.google.com/sdk/docs/quickstart-linux)

4. Run gcp-restart in non-preemptible VM
```sh
nohup gcp-restart &> log.gcp-restart &
```
