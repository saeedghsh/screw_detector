## CVAT
### CVAT: install
Installed by the instructions from [this page](https://docs.cvat.ai/docs/administration/basics/installation/).  
```bash
# docker and docker compose was already installed and docker group created, as saeed user added to group
git clone https://github.com/cvat-ai/cvat
cd cvat
```

### CVAT: launch local server
```bash
docker compose up -d
```

### CVAT: register user
Register a user with the [local] cvat server:
```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

### CVAT: access UI
Go to `localhost:8080` and log-in using the credentials above. CVAT claims that
only chrome is supported, but I tested with Firefox and it was working fine!

### CVAT: shutdown
To Shutdown:
```bash
cd cvat
docker compose down
```

### CVAT: pointcloud file type
CVAT only supports `.pcd` and `.bin`, not `.ply`! So a simple script was created
to convert `.ply` files to `.pcd`.

### CVAT: issue, not rendering 3D data
Initially it was assumed (by ChatGPT) that redis and memory was the issue. Tried
so many different things before I gave up on resolving it (due to limited
resources and the very near deadline).

One possible solution we tried was to enable "Memory Overcommitment" (as shown
below) and relaunch. But it did not help!
* enable memory overcommitment
  * Temporarily enable memory overcommitment (until reboot):
    `sudo sysctl vm.overcommit_memory=1`
  * Permanently enable it by adding the following line to `/etc/sysctl.conf`:
    `vm.overcommit_memory = 1`
* Apply the change:
  `sudo sysctl -p`
* `docker compose down`
* `docker compose up -d`

Another solution we tried was to launch chrome with increased memory, which
didn't help either.

### CVAT: conclusion
Finally, after trying many tricks suggested by chatgpt, we could not resolve the
issue. Decided to use CVAT only for 2D and use Open3D for 3D vis. Since we have
the transformation between pointcloud and camera origin, we should be able to
transfer 2D annotation to 3D (to some extend!).

